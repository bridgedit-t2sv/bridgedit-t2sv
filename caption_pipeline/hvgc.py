from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch
import os
import json
from tqdm import tqdm
import argparse
# 设置 NCCL_DEBUG 环境变量为 ERROR，禁用详细日志
os.environ["NCCL_DEBUG"] = "ERROR"

step1_template = """You are an expert video analyst and prompt engineer. Your goal is to watch a video and create a highly effective, descriptive prompt that can be used by a text-to-video generation model to recreate the visual essence and physical dynamics of the scene.
Your generated prompt should be a dense, continuous paragraph rich with visual details. Follow this thinking process:
1.  **Scene and Atmosphere**:
    * Describe the core environment and the overall mood. Translate any auditory feeling (e.g., tension from music) into visual terms (e.g., `high-contrast lighting, deep shadows`).
2.  **Subjects and Details**:
    * Identify the main subjects and objects. Describe them with specific visual adjectives (`a weathered blacksmith with soot-stained hands`, `a glowing orange piece of iron`).
3.  **Key Actions, Cinematography, and Physical Dynamics**:
    * Describe the sequence of most important actions.
    * Describe the cinematography (shot type, angle, movement).
    * **[NEW CORE RULE] Describe the "Visual Counterpart" of Sound**: Instead of describing sound itself, describe the physical actions that *create* sound. Focus on impact, interaction, and motion that implies sound.
        * **For Speech/Vocalization**: Detail the mouth movements, facial expressions, and throat or chest movements (`a lion opens its massive jaws wide, a deep roar building in its chest`).
        * **For Impacts**: Describe the collision, the reaction, and the result (`a heavy hammer strikes the glowing iron, sending a shower of bright orange sparks flying into the air; the metal visibly deforms under the blow`).
        * **For Movement/Friction**: Detail the interaction between surfaces (`a car's tires screech, leaving black rubber marks on the asphalt as it drifts around a corner`).
        * **For Natural Forces**: Describe the effect of the force on the environment (`trees bend and sway violently under the force of the wind, loose leaves are whipped into a frenzy`).
4.  **Visual Style and Quality**:
    * Specify the artistic style (`photorealistic`, `cinematic`), lighting (`dramatic, warm light from the forge`), and visual quality (`highly detailed, 8K`).
**Final Instruction**: Synthesize all these visual and physical elements into a single, rich, and coherent paragraph. Your entire output should be a prompt that visually and dynamically directs an AI. **Do NOT describe sound itself, but rather the physics of its creation.**
Based on the video, the prompt for video generation is:"""

step2_template = """You are an expert AI assistant specializing in identifying auditory concepts from text. Your task is to read a descriptive video caption and extract a list of the primary objects or events that produce sound.
Instructions:
Analyze the provided caption to understand the scene.
Identify only the key elements that would create distinct sounds.
Ignore general scenery, static objects, or characters not directly involved in making a sound.
Output the concepts as a simple, comma-separated list of keywords.
Example:
Input Caption: A strong blacksmith forcefully strikes a hammer on glowing iron in the workshop, sending a shower of sparks.
Audio Labels: Hammer, Sparks
Your Task:
Input Caption: {video_caption}
Audio Labels:"""

step3_template = """
### TASK
Generate a concise, descriptive audio-only caption. You will be given a detailed video caption for context and a short list of key audio labels. This generated caption will be used as a prompt for a text-to-audio model, so it must describe a single, co-existing soundscape.
### GUIDELINES
1.  **Audio Only**: The caption MUST describe only the audible aspects. Do not mention or allude to any visual information from the video caption.
2.  **No Sequences**: DO NOT use temporal words (e.g., "first," "then," "starts with," "followed by"). Describe all sounds as if they are happening together in one scene.
3.  **Describe Characteristics**: Based on the labels, describe the sound's quality.
    * **Human Sounds**: e.g., "a low-pitched voice," "a loud shout."
    * **Sound Effects**: e.g., "a door creaking," "a hammer striking metal."
    * **Music**: e.g., "an upbeat electronic drum beat," "a slow acoustic guitar."
    * **Environment**: e.g., "the rumble of city traffic," "wind howling," "birds chirping."
4.  **Final Output**: The result must be a single, continuous audio caption that combines the elements from the audio labels into one coherent phrase.
---
### EXAMPLE
#### Input Context:
* **Video Caption**: A young man with a red backpack walks briskly down a paved path in a large city park during autumn. The wind is strong, whipping through the nearly bare trees and rustling his jacket. He wears large, black headphones, nodding his head to a beat. A small terrier on a leash ahead of him stops suddenly and lets out a sharp bark at a squirrel.
* **Audio Labels**: `["Strong Wind", "Rock Music with Electric Guitar", "Dog Bark"]`
#### Correct Output:
* **Audio Caption**: The powerful roar of strong wind, a sharp dog bark, and the driving sound of rock music with an electric guitar all blend together in a dynamic soundscape.
---
### YOUR TASK
**Video Caption**: {video_caption}
**Audio Labels**: ["{audio_labels}"]
**Audio Caption**:"""

# MODEL_PATH = "/mnt/task_runtime/models/Qwen2.5-VL-72B-Instruct"
def load_mllm_model(model_path=MODEL_PATH):
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        limit_mm_per_prompt={"image": 10, "video": 10},
        dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return llm, processor

def load_llm(model_path=MODEL_PATH):
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=8,
        dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return llm, tokenizer
   
def process_video_input(video_files_list, processor):
    video_inputs_list = []
    # print("Processing video message")
    for video_file in video_files_list:
        video_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                    {"type": "text", "text": step1_template},
                    {
                        "type": "video", 
                        "video": video_file,
                        "total_pixels": 1280 * 28 * 28, "min_pixels": 16 * 28 * 28
                    }
                ]
            },
        ]
        prompt = processor.apply_chat_template(
            video_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        _, video_inputs, video_kwargs = process_vision_info(video_messages, return_video_kwargs=True)
        # Check and remove unnecessary keys from video_kwargs
        mm_data = {}
        mm_data["video"] = video_inputs
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        video_inputs_list.append(llm_inputs)
    return video_inputs_list

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=1024,
    stop_token_ids=[],
)

def generate_video_caption(video_path=None, batch_size=1, mllm=None, processor=None):
    video_files_list = [os.path.join(video_path, file) for file in os.listdir(video_path)]
    video_js = {}
    for video_file in video_files_list:
        video_js[video_file] = {}
    video_files_list = list(video_js.keys())
    print("Lens of video_js: ", len(video_js))
    for i in tqdm(range(0, len(video_files_list), batch_size)):
        video_batch_inputs = process_video_input(video_files_list[i:i+batch_size], processor)
        outputs = llm.generate(video_batch_inputs, sampling_params=sampling_params, use_tqdm=False)
        gen_captions = [output.outputs[0].text for output in outputs]
        for j in range(batch_size):
            try:
                video_id = video_files_list[i+j]
                video_js[video_id]['video_caption'] = gen_captions[j]
            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                del video_js[video_id]
                continue
    # with open('recaption/landscape-captions-test.json','w') as f:
    #     json.dump(video_js, f, indent=4)
    return video_js
    
def generate_audio_labels(video_js, batch_size=1, llm=None, tokenizer=None):
    keys_list = list(video_js.keys())
    for i in tqdm(range(0, len(video_files_list), batch_size)):
        batch_keys_list = keys_list[i : i+batch_size]
        video_captions = [caption_js[key]['video_caption'] for key in batch_keys_list]
        label_inputs = []
        for idx,video_caption in enumerate(video_captions):
            audio_messages = [
                    {"role": "system", "content": "You are a helpful assisstant for caption."},
                    {"role": "user", "content": step2_template.format(
                            video_caption=video_caption
                        )
                    }
                ]
            audio_input = tokenizer.apply_chat_template(
                audio_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            audio_inputs.append(audio_input)
        audio_label_outputs = llm.generate(audio_inputs, sampling_params, use_tqdm=False)
        audio_labels = [output.outputs[0].text for output in audio_label_outputs]
        for j in range(batch_size):          
            video_id = batch_keys_list[j]
            video_js[video_id]['labels'] = audio_labels[j]     
    return video_js

def generate_audio_caption(video_js, batch_size=1, llm=None, tokenizer=None):
    keys_list = list(video_js.keys())
    for i in tqdm(range(0, len(video_files_list), batch_size)):
        batch_keys_list = keys_list[i : i+batch_size]
        video_captions = [caption_js[key]['video_caption'] for key in batch_keys_list]
        label_inputs = []
        for idx,video_caption in enumerate(video_captions):
            audio_messages = [
                    {"role": "system", "content": "You are a helpful assisstant for caption."},
                    {"role": "user", "content": step2_template.format(
                            video_caption=video_caption,
                            audio_labels=video_js[batch_keys_list[idx]]['labels']
                        )
                    }
                ]
            audio_input = tokenizer.apply_chat_template(
                audio_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            audio_inputs.append(audio_input)
        audio_caption_outputs = llm.generate(audio_inputs, sampling_params, use_tqdm=False)
        audio_captions = [output.outputs[0].text for output in audio_caption_outputs]
        for j in range(batch_size):
            video_id = batch_keys_list[j]
            video_js[video_id]['audio_caption'] = audio_captions[j]
    return video_js
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="/mnt/task_runtime/dataset/vgg-ss/video")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mllm_path", type=str, default="/mnt/task_runtime/models/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--llm_path", type=str, default="/mnt/task_runtime/models/Qwen2.5-72B-Instruct")
    parser.add_argument("--output_file", type=str, default="recaption/vgg-ss-captions.json")
    args = parser.parse_args()

    mllm_path = args.mllm_path
    llm_path = args.llm_path
    output_file = args.output_file
    batch_size = args.batch_size

    video_path = args.video_path
    mllm, processor = load_mllm_model(mllm_path)
    video_js =generate_video_caption(mllm=mllm,processor=processor, video_path=video_path, batch_size=batch_size)
    del mllm, processor
    llm, tokenizer = load_llm(llm_path)
    video_js_with_audio_labels = generate_audio_labels(video_js, batch_size=batch_size, llm=llm, tokenizer=tokenizer)
    video_js_with_audio_caption = generate_audio_caption(video_js_with_audio_labels, batch_size=batch_size, llm=llm, tokenizer=tokenizer)
    with open(output_file, 'w') as f:
        json.dump(video_js_with_audio_caption, f, indent=4)


