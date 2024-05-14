import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering

# Önceden eğitilmiş model ve tokenizer'ı yükleme
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


def predict_answer(image_path, question):
    # Görüntüyü yükleme ve ön işleme
    image = Image.open(image_path)
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = image_transform(image)

    # Soruyu tokenlara dönüştürme ve özel işaretleme eklemek
    question_tokens = tokenizer.encode("[CLS] " + question + " [SEP]", add_special_tokens=True)

    # Model için giriş oluşturma
    input_ids = torch.tensor([question_tokens])
    image = torch.unsqueeze(image, 0)  # Batch boyutunu ekleyin
    input_ids = input_ids.unsqueeze(0)  # Batch boyutunu ekleyin

    # Modelden cevapları alın
    outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), 
                    image=image.unsqueeze(0), return_dict=True)

    # Cevapların başlangıç ve bitiş indekslerini alın
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)

    # Cevapları tokenlerden metne dönüştürme
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end+1]))

    return answer
