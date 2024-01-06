import argparse
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import OrderedDict

def get_input_args():
    """
    function that gets command line arguments
    .............
    Output:
        command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', type=str, default='flower_data/', help="path to directory")
    parser.add_argument('--save_dir', type=str, default='', help='Set directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13', help='Choose architecture')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='sets learning rate')
    parser.add_argument('--hidden_unites', type=int, default=512, help='sets the number of hidden unites in each hidden layer')
    parser.add_argument('--epochs', type=int, default=7, help='sets the number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    in_args = parser.parse_args()

    return in_args


class Data:
    """
    Class for handling data loading and transformation
    .............
    Methods:
        __init__: Initializes the Data class
        get_trainloader: Returns the train data loader
        get_testloader: Returns the test data loader
        get_validloader: Returns the validation data loader
    """
    def __init__(self, train_dir, valid_dir, test_dir, batch_size=128) -> None:
        """
        Initializes the Data class with directories for train, validation, and test datasets
        .............
        Input:
            train_dir: Directory path to the training dataset
            valid_dir: Directory path to the validation dataset
            test_dir: Directory path to the test dataset
            batch_size: Batch size for data loaders (default: 128)
        """
        # Code for setting up data loaders
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
        test_valid_transform = transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])])
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
        self.test_data = datasets.ImageFolder(test_dir, transform=test_valid_transform)
        self.valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transform)
        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size)
        self.validloader = torch.utils.data.DataLoader(self.valid_data, batch_size=batch_size)


    def get_trainloader(self):
        """
        Returns the train data loader
        .............
        Output:
            Train data loader
        """
        return self.trainloader
    

    def get_testloader(self):
        """
        Returns the test data loader
        .............
        Output:
            Test data loader
        """
        return self.testloader
    

    def get_validloader(self):
        """
        Returns the validation data loader
        .............
        Output:
            Validation data loader
        """
        return self.validloader
    

class Model:
    """
    Class for building and training the model
    .............
    Methods:
        __init__: Initializes the Model class
        build_classifier: Builds the classifier for the model
        prepare_model: Prepares the model with specified architecture
        training_device: Returns the training device (CPU/GPU)
        test_model: Evaluates the model on test data
        train_model: Trains the model
        save_model: Saves the trained model
    """
    def __init__(self, learning_rate: float, epochs: int, arch: str, save_dir: str, hidden_units: int, gpu: bool, data: Data) -> None:
        """
        Initializes the Model class with model parameters and data
        .............
        Input:
            learning_rate: Learning rate for training the model
            epochs: Number of epochs for training
            arch: Architecture of the model (e.g., 'vgg13', 'vgg16')
            save_dir: Directory to save the trained model
            hidden_units: Number of hidden units in the classifier
            gpu: Boolean flag indicating GPU usage
            data: Data object containing train, validation, and test loaders
        """
        # Code for setting up the model
        self.data = data
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.arch = arch
        self.save_dir = save_dir
        self.hidden_units = hidden_units
        self.gpu = gpu
        self.model = self.prepare_model()
        self.model.classifier = self.build_classifier()
        self.criterion = torch.nn.NLLLoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.model.to(self.device)
        


    def build_classifier(self):
        """
        Builds the classifier for the model
        .............
        Output:
            Model classifier
        """
        # Code for building the classifier for the model
        classifier = torch.nn.Sequential(OrderedDict([('fc1', torch.nn.Linear(25088, self.hidden_units)),
                                                      ('relu1', torch.nn.ReLU()),
                                                      ('drop1', torch.nn.Dropout(0.1)),
                                                      ('fc2', torch.nn.Linear(self.hidden_units, self.hidden_units)),
                                                      ('relu2', torch.nn.ReLU()),
                                                      ('drop1', torch.nn.Dropout(0.1)),
                                                      ('fc3', torch.nn.Linear(self.hidden_units, 102)),
                                                      ('output', torch.nn.LogSoftmax(dim=1))]))
        return classifier

    def prepare_model(self):
        """
        Prepares the model with specified architecture
        .............
        Output:
            Pretrained model with specified architecture
        """
        # Code for preparing the model with specified architecture
        model = torch.hub.load('pytorch/vision', self.arch, pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    def training_device(self):
        """
        Returns the training device (CPU/GPU)
        .............
        Output:
            Training device (CPU/GPU)
        """
        return self.device
    
    def test_model(self, training_loss):
        """
        Evaluates the model on test data
        .............
        Input:
            training_loss: Loss value from the training phase
        """
        with torch.no_grad():
            self.model.eval()
            accuracy = 0
            test_loss = 0
            for images, labels in tqdm(self.data.get_testloader()):
                images, labels = images.to(self.device), labels.to(self.device)
                
                log_ps = self.model(images)
                
                loss = self.criterion(log_ps, labels)
                test_loss += loss.item()
                
                top_p, top_class = log_ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
            print('accuracy: {:0.2%}'.format(accuracy/len(self.data.get_testloader())))
            print('Training loss: {}'.format(training_loss/len(self.data.get_trainloader())))
            print('Test_loss: {}'.format(test_loss/len(self.data.get_trainloader())))
            print('-------------------------------------------------------------')
        self.model.train()

    def train_model(self):
        """
        Trains the model
        """
        for epoch in range(self.epochs):
            training_loss = 0
            print('epoch {}/{}'.format(epoch+1, self.epochs))
            for images, labels in tqdm(self.data.get_trainloader()):
                images, labels = images.to(self.device), labels.to(self.device)
                
                log_ps = self.model(images)
                loss = self.criterion(log_ps, labels)
        
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()
            else:
                self.test_model(training_loss)
    
    def save_model(self):
        """
        Saves the trained model
        """
        checkpoint = {'arch': self.arch,
                      'classifier': self.model.classifier,
                      'state_dict': self.model.state_dict()}
        if self.save_dir[-1] != '/':
            self.save_dir += '/'
        torch.save(checkpoint, self.save_dir + self.arch + str(self.hidden_units) +'checkpoint.pth')


def main():
    """
    Main function to run the model training
    .............
    """
    archs = ['vgg13', 'vgg16']
    in_args = get_input_args()

    data_dir = in_args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    learning_rate = in_args.learning_rate
    arch = in_args.arch
    hidden_units = in_args.hidden_unites
    epochs = in_args.epochs
    if arch not in archs:
        print('only vgg13 and vgg16 models are permissible.')
        raise ValueError

    save_dir = in_args.save_dir
    gpu = in_args.gpu

    data = Data(train_dir, valid_dir, test_dir)
    model = Model(learning_rate, epochs, arch, save_dir, hidden_units, gpu, data)

    print('model training on: --{}--'.format(model.training_device()))
    model.train_model()
    model.save_model()


if __name__ == '__main__':
    main()