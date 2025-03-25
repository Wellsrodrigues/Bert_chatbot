from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="pierreguillou/bert-base-cased-squad-v1.1-portuguese")

with open("info.txt", "r", encoding="utf-8") as file:
    context = file.read()


if __name__ == "__main__":
    print("Chatbot sobre Alzheimer.\nDigite sua pergunta ou 'sair' para encerrar.")
    
    while True:
        question = input("Você: ")
        if question.lower() == 'sair':
            break
        
        response = qa_pipeline(question=question, context=context)
        print("Chatbot:", response["answer"])