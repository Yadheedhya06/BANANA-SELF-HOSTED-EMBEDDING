from simpletransformers.language_representation import RepresentationModel
import torch
import json


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    model = RepresentationModel(
        model_type="bert",
        model_name="bert-base-uncased",
        use_cuda= False
    )

# # Inference is ran for every server call
# # Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    return {'message': "test"}
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    word_vectors = model.encode_sentences(prompt, combine_strategy=None)
    # result = json.dumps(word_vectors)

    # return {'message': "test"}



# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain.embeddings import SelfHostedEmbeddings
# import runhouse as rh
# import torch


# gpu = rh.cluster(name="rh-a10x", instance_type="A100:1", use_spot=False)

# text = input("Enter text:")

# def get_pipeline():
#     model_id = "bert-base-uncased"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(model_id)
#     return pipeline("Fill-Mask", model=model, tokenizer=tokenizer)

# def inference_fn(pipeline, prompt):
#     # Return last hidden state of the model
#      if isinstance(prompt, list):
#         return [emb[0][-1] for emb in pipeline(prompt)] 
#      return pipeline(prompt)[0][-1]

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
# def init():
#     global model
    
#     device = 0 if torch.cuda.is_available() else -1
#     model = pipeline('fill-mask', model='bert-base-uncased', device=device)


# # Inference is ran for every server call
# # Reference your preloaded global model variable here.
# def inference(model_inputs:dict) -> dict:
#     global model

#     # Parse out your arguments
#     prompt = model_inputs.get('prompt', None)
#     if prompt == None:
#         return {'message': "No prompt provided"}
    
#     # Run the model
#     gpu = rh.cluster(name="rh-a10x", instance_type="A100:1", use_spot=False)

#     model_id = "bert-base-uncased"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(model_id)
#     pipe =  pipeline("Fill-Mask", model=model, tokenizer=tokenizer)

#     if isinstance(prompt, list):
#         infr =  [emb[0][-1] for emb in pipeline(prompt)] 
#     infr =  pipeline(prompt)[0][-1]

#     embeddings = SelfHostedEmbeddings(
#     model_load_fn=pipe, 
#     hardware=gpu,
#     model_reqs=["./", "torch", "transformers"],
#     inference_fn=infr)

#     # result = embeddings.embed_query(prompt)
#     # Return the results as a dictionary
#     # result = {'output': embeddings.embed_query(prompt)}
#     return {embeddings.embed_query(prompt)}

# if __name__ == "__main__":
#     init()
#     print(inference({'prompt': text}))
