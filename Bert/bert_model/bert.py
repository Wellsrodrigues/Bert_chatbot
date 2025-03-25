import torch, json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def load_model():
    model_path = "Khanh/bert-base-multilingual-cased-finetuned-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    return tokenizer, model

def answer_question(question, context, tokenizer, model):
    inputs = tokenizer(question, context, return_tensors='pt', truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end])
    )
    
    return answer.strip()

def clean_response(response):
    special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"]
    for token in special_tokens:
        response = response.replace(token, "")
    
    response = response.strip()
    
    if not response:
        return "Não tenho uma resposta específica para isso."
    
    return response

def load_json_dataset(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['pergunta']: item['resposta'] for item in data}

def main():
    tokenizer, model = load_model()
    dataset = load_json_dataset("dataset.json")
    
    print("Chatbot sobre Alzheimer. Digite sua pergunta ou 'sair' para encerrar.")
    while True:
        question = input("Você: ")
        if question.lower() == 'sair':
            break
        
        context = dataset.get(question, "Não tenho essa informação no momento.")
        response = answer_question(question, context, tokenizer, model)
        response = clean_response(response)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
