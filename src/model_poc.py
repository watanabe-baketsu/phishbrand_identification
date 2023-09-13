import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, ViTFeatureExtractor, ViTModel


class MultiModalT5(nn.Module):
    def __init__(self, text_model_name='t5-small', vision_model_name='google/vit-base-patch16-224-in21k'):
        super(MultiModalT5, self).__init__()

        self.text_model = T5ForConditionalGeneration.from_pretrained(text_model_name)
        self.vision_model = ViTModel.from_pretrained(vision_model_name)

        self.tokenizer = T5Tokenizer.from_pretrained(text_model_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(vision_model_name)

        self.linear = nn.Linear(self.text_model.config.d_model + self.vision_model.config.hidden_size,
                                self.text_model.config.d_model)

    def forward(self, text_input, image_input):
        # Process text
        text_input = self.tokenizer(text_input, return_tensors='pt', padding=True, truncation=True)
        text_output = self.text_model(**text_input).last_hidden_state

        # Process image
        image_input = self.feature_extractor(image_input)['pixel_values']
        vision_output = self.vision_model(image_input).last_hidden_state
        vision_output = vision_output.mean(dim=1)

        # Concatenate and project
        combined = torch.cat((text_output[:, 0, :], vision_output), dim=1)
        projected = self.linear(combined)

        # Generate text using T5
        generated_output = self.text_model.generate(encoder_outputs=(projected.unsqueeze(0),), max_length=50)

        # Decode output
        decoded_output = self.tokenizer.decode(generated_output[0])

        return decoded_output


# Instantiate the model
model = MultiModalT5()

# Dummy input
text_input = ["This is some HTML content"]
image_input = torch.randn(1, 3, 224, 224)  # Assuming the image is preprocessed to fit ViT

# Forward pass
output = model(text_input, image_input)
print(output)
