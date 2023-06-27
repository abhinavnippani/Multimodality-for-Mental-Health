
import torch
from torch import nn
import torch.nn.functional as F

from models.fusions import *


#------------------------------- Unimodal Img ------------------------------------------#

class UnimodalImg(nn.Module):
    """
    PyTorch module for the Unimodal Image model.

    Args:
        modality1 (nn.Module): The first modality model.
        modality2 (nn.Module): The second modality model.
        configuration (dict): Configuration parameters for the model.
        device (torch.device): The device on which the model will be run.

    Returns:
        torch.Tensor: The output tensor of the model.

    """

    def __init__(self, modality1, modality2, configuration, device):
        super().__init__()
        self.modality1 = modality1
        self.modality2 = modality2
        self.config = configuration
        self.device = device

        self.head = MLP(in_channels=self._calculate_in_features(),
                        num_classes=self.config['Models']['mlp_num_classes'],
                        hidden_sizes=self.config['Models']['mlp_hidden_sizes'],
                        dropout_probability=self.config['Models']['mlp_dropout_prob'],
                        user=self.config['General']['user'])

        if self.config['Models']['encoder_finetuning'] == False:
            # Freeze the parameters of modality1 and modality2 if encoder finetuning is disabled
            for param in self.modality1.parameters():
                param.requires_grad = False

            for param in self.modality2.parameters():
                param.requires_grad = False

        for param in self.head.parameters():
            param.requires_grad = True

    def forward(self, input1, input2):
        """
        Forward pass of the Unimodal Image model.

        Args:
            input1 (Tensor): The input tensor for modality 1.
            input2: The input tensor for modality 2.

        Returns:
            Tensor: The output tensor of the model.

        """
        if self.config['General']['user']:
            img_processor = self.config['Models']['image_processor']
            image_output = []

            for img in input1:
                img_processed = img_processor(img, return_tensors='pt').to(self.device)
                img_output = self.modality1(**img_processed)['last_hidden_state'].to(self.device)
                img_output = torch.mean(img_output, 1).to(self.device)
                image_output.append(img_output)

            image_output = torch.cat(image_output, dim=0)
            image_output = torch.mean(image_output, dim=0, keepdim=True)

        else:
            image_output = self.modality1(**input1)['last_hidden_state'].to(self.device)
            image_output = torch.mean(image_output, 1).to(self.device)

        head_output = self.head.to(self.device)(image_output).to(self.device)
        return head_output

    def _calculate_in_features(self):
        """
        Calculate the number of input features for the head model.

        Returns:
            int: The number of input features.

        """
        # Create an example input and pass it through the network to get the output size
        img_batch = torch.randint(0, 255, size=(self.config['General']['batch_size'], 3, self.config['General']['img_size'], self.config['General']['img_size'])).float()
        img_processor = self.config['Models']['image_processor']
        input1 = img_processor(img_batch, return_tensors='pt').to(self.device)
        image_output = self.modality1(**input1)['last_hidden_state'].to(self.device)
        image_output = torch.mean(image_output, 1).to(self.device)
        return image_output.shape[-1]

#------------------------------- Unimodal Txt ------------------------------------------#

class UnimodalTxt(nn.Module):
    """
    PyTorch module for the Unimodal Text model.

    Args:
        modality1 (nn.Module): The first modality model.
        modality2 (nn.Module): The second modality model.
        configuration (dict): Configuration parameters for the model.
        device (torch.device): The device on which the model will be run.

    Returns:
        torch.Tensor: The output tensor of the model.

    """

    def __init__(self, modality1, modality2, configuration, device):
        super().__init__()
        self.modality1 = modality1
        self.modality2 = modality2
        self.config = configuration
        self.device = device

        self.head = MLP(in_channels=self._calculate_in_features(),
                        num_classes=self.config['Models']['mlp_num_classes'],
                        hidden_sizes=self.config['Models']['mlp_hidden_sizes'],
                        dropout_probability=self.config['Models']['mlp_dropout_prob'],
                        user=self.config['General']['user'])

        if self.config['Models']['encoder_finetuning'] == False:
            # Freeze the parameters of modality1 and modality2 if encoder finetuning is disabled
            for param in self.modality1.parameters():
                param.requires_grad = False

            for param in self.modality2.parameters():
                param.requires_grad = False

        for param in self.head.parameters():
            param.requires_grad = True

    def forward(self, input1, input2):
        """
        Forward pass of the Unimodal Text model.

        Args:
            input1 (Tensor): The input tensor for modality 1.
            input2: The input tensor for modality 2.

        Returns:
            Tensor: The output tensor of the model.

        """
        if self.config['General']['user']:
            tokenizer_txt = self.config['Models']['text_tokenizer']

            text_output = []
            for txt in input2[:64]:
                txt_processed = tokenizer_txt(txt, return_tensors='pt', padding=True, truncation=True).to(self.device)
                txt_output = self.modality2(**txt_processed)['last_hidden_state'].to(self.device)
                txt_output = torch.mean(txt_output, 1).to(self.device)
                text_output.append(txt_output)

            text_output = torch.cat(text_output, dim=0)
            text_output = torch.mean(text_output, dim=0, keepdim=True)

        else:
            text_output = self.modality2(**input2)['last_hidden_state'].to(self.device)
            text_output = torch.mean(text_output, 1).to(self.device)

        head_output = self.head.to(self.device)(text_output).to(self.device)

        return head_output

    def _calculate_in_features(self):
        """
        Calculate the number of input features for the head model.

        Returns:
            int: The number of input features.

        """
        # Create an example input and pass it through the network to get the output size
        tokenizer_txt = self.config['Models']['text_tokenizer']
        text_batch = "This is a sample input for shape inference"
        text_batch = [text_batch] * self.config['General']['batch_size']
        input2 = tokenizer_txt(text_batch, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # Forward pass until MLP
        text_output = self.modality2(**input2)['last_hidden_state'].to(self.device)
        text_output = torch.mean(text_output, 1).to(self.device)

        return text_output.shape[1]

#------------------------------- Late Fusion ------------------------------------------#

class LateFusion(nn.Module):
    """
    PyTorch module for the Late Fusion model.

    Args:
        modality1 (nn.Module): The first modality model.
        modality2 (nn.Module): The second modality model.
        configuration (dict): Configuration parameters for the model.
        device (torch.device): The device on which the model will be run.

    Returns:
        torch.Tensor: The output tensor of the model.

    """

    def __init__(self, modality1, modality2, configuration, device):
        super().__init__()
        self.modality1 = modality1
        self.modality2 = modality2
        self.config = configuration
        self.device = device

        self.head = MLP(in_channels=self._calculate_in_features(),
                        num_classes=self.config['Models']['mlp_num_classes'],
                        hidden_sizes=self.config['Models']['mlp_hidden_sizes'],
                        dropout_probability=self.config['Models']['mlp_dropout_prob'],
                        user=self.config['General']['user'])

        if self.config['Models']['encoder_finetuning'] == False:
            # Freeze the parameters of modality1 and modality2 if encoder finetuning is disabled
            for param in self.modality1.parameters():
                param.requires_grad = False

            for param in self.modality2.parameters():
                param.requires_grad = False

        for param in self.head.parameters():
            param.requires_grad = True

    def forward(self, input1, input2):
        """
        Forward pass of the Late Fusion model.

        Args:
            input1 (Tensor): The input tensor for modality 1.
            input2: The input tensor for modality 2.

        Returns:
            Tensor: The output tensor of the model.

        """
        if self.config['General']['user']:
            img_processor = self.config['Models']['image_processor']
            tokenizer_txt = self.config['Models']['text_tokenizer']

            image_output = []
            text_output = []

            for img in input1:
                img_processed = img_processor(img, return_tensors='pt').to(self.device)
                img_output = self.modality1(**img_processed)['last_hidden_state'].to(self.device)
                img_output = torch.mean(img_output, 1).to(self.device)
                image_output.append(img_output)

            image_output = torch.cat(image_output, dim=0)
            image_output = torch.mean(image_output, dim=0, keepdim=True)

            for txt in input2:
                txt_processed = tokenizer_txt(txt, return_tensors='pt', padding=True, truncation=True).to(self.device)
                txt_output = self.modality2(**txt_processed)['last_hidden_state'].to(self.device)
                txt_output = torch.mean(txt_output, 1).to(self.device)
                text_output.append(txt_output)

            text_output = torch.cat(text_output, dim=0)
            text_output = torch.mean(text_output, dim=0, keepdim=True)

        else:
            image_output = self.modality1(**input1)['last_hidden_state'].to(self.device)
            image_output = torch.mean(image_output, 1).to(self.device)

            text_output = self.modality2(**input2)['last_hidden_state'].to(self.device)
            text_output = torch.mean(text_output, 1).to(self.device)

        fusion_output = Concat().to(self.device)([image_output, text_output]).to(self.device)

        head_output = self.head.to(self.device)(fusion_output).to(self.device)

        return head_output

    def _calculate_in_features(self):
        """
        Calculate the number of input features for the head model.

        Returns:
            int: The number of input features.

        """
        # Create example inputs and pass them through the network to get the output size
        img_batch = torch.randint(0, 255, size=(self.config['General']['batch_size'], 3, self.config['General']['img_size'], self.config['General']['img_size'])).float()
        img_processor = self.config['Models']['image_processor']
        tokenizer_txt = self.config['Models']['text_tokenizer']
        text_batch = "This is a sample input for shape inference"
        text_batch = [text_batch] * self.config['General']['batch_size']
        input1 = img_processor(img_batch, return_tensors='pt').to(self.device)
        input2 = tokenizer_txt(text_batch, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # Forward pass until MLP
        image_output = self.modality1(**input1)['last_hidden_state'].to(self.device)
        image_output = torch.mean(image_output, 1).to(self.device)
        text_output = self.modality2(**input2)['last_hidden_state'].to(self.device)
        text_output = torch.mean(text_output, 1).to(self.device)
        fusion_output = Concat().to(self.device)([image_output, text_output]).to(self.device)

        return fusion_output.shape[1]

#-------------------------------- MLP ------------------------------------------#

class MLP(nn.Module):
    """
    PyTorch module for a Multi-Layer Perceptron (MLP) model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        hidden_sizes (list): List of hidden layer sizes.
        dropout_probability (list): List of dropout probabilities for each hidden layer.
        user (bool): Indicates whether the model is used for user classification.

    Returns:
        torch.Tensor: The output tensor of the model.

    """

    def __init__(self, in_channels, num_classes, hidden_sizes=[128, 64], dropout_probability=[0.5, 0.7], user=False):
        super(MLP, self).__init__()
        assert len(hidden_sizes) >= 1, "Specify at least one hidden layer"

        self.layers = self.create_layers(in_channels, num_classes, hidden_sizes, dropout_probability, user)

    def create_layers(self, in_channels, num_classes, hidden_sizes, dropout_probability, user=False):
        """
        Create the layers of the MLP model.

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            hidden_sizes (list): List of hidden layer sizes.
            dropout_probability (list): List of dropout probabilities for each hidden layer.
            user (bool): Indicates whether the model is used for user classification.

        Returns:
            nn.Sequential: The sequential container for the MLP layers.

        """
        layers = []
        layer_sizes = [in_channels] + hidden_sizes + [num_classes]
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                if user:
                    layers.append(nn.InstanceNorm1d(layer_sizes[i+1]))
                else:
                    layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_probability[i]))
            else:
                layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MLP model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor of the model.

        """
        out = x.view(x.shape[0], -1)
        out = self.layers(out)
        return out
