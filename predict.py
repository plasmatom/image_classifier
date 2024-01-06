import argparse
import torch
import json
import PIL
import numpy as np

def get_input_args():
    """
    function that gets command line arguments
    .............
    Output:
        command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type=str, default='flower_data/', help="path to image")
    parser.add_argument('check_point', type=str, default='checkpoint.pth', help='path to model check-point')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help=' Use a mapping of categories to real names:')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    in_args = parser.parse_args()

    return in_args


class Predictor:
    """
    Class for making predictions based on a pre-trained model
    .............
    Methods:
        __init__: Initializes the Predictor class
        get_categories: Retrieves categories from a JSON file
        rebuild_model: Rebuilds a pre-trained model
        process_image: Preprocesses the input image for prediction
        predict: Makes predictions on the input image
    """
    def __init__(self, image_path,check_point, top_k, category_path, gpu) -> None:
        # Initialization of Predictor class with provided arguments
        # Code for setting up class attributes
        self.image_path = image_path
        self.check_point = check_point
        self.top_k = top_k
        self.cat_to_name = self.get_categories(category_path)
        self.class_names = sorted([label for label, cls in self.cat_to_name.items()])
        self.model = self.rebuild_model()

        if gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

    def get_categories(self, path):
        """
        Method to get categories from a JSON file
        .............
        Input:
            path: Path to the JSON file containing categories
        Output:
            Dictionary containing categories
        """
        # Code for loading categories from a JSON file
        with open(path, 'r') as f:
            cat_to_name = json.load(f)
        return cat_to_name
    
    def rebuild_model(self):
        """
        Method to rebuild a pre-trained model using a checkpoint
        .............
        Output:
            Rebuilt pre-trained model
        """
        # Code for rebuilding a pre-trained model using a checkpoint
        state_dict = torch.load(self.check_point)
        model = torch.hub.load('pytorch/vision', state_dict['arch'], pretrained=True)
        model.classifier = state_dict['classifier']
        model.load_state_dict(state_dict['state_dict'])
        return model
    
    def process_image(self, image):
        """
        Method to preprocess the image for prediction
        .............
        Input:
            image: Input image to be preprocessed
        Output:
            Processed image ready for prediction
        """
        # Code for preprocessing the input image
        with PIL.Image.open(image) as im:
            width, height = im.size
            if width < height:
                height = int(height / width * 256)
                width = 256
            else:
                width = int(width / height * 256)
                height = 256
                
            im = im.resize((width, height), PIL.Image.LANCZOS)

            left = (width - 224) / 2
            top = (height - 224) / 2
            right = (width + 224) / 2
            bottom = (height + 224) / 2
            img =im.crop((left, top, right, bottom))
            

        np_image =np.array(img)
        
        np_image = np_image / 255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std
        
        return np_image.transpose((2, 0, 1))
    
    def predict(self):
        """
        Method to make predictions on the image
        .............
        Output:
            Top K probabilities and predicted classes
        """
        # Code for making predictions on the image
        image = torch.Tensor(self.process_image(self.image_path)).to(self.device)
        image = image.view(1, *image.shape)
        self.model.to(self.device)
        
        log_ps = self.model(image)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(self.top_k, dim=1)
        top_p = top_p[0, :].tolist()
        top_class = top_class[0, :].tolist()
        torch.cuda.empty_cache()
        
        return top_p, [self.cat_to_name[self.class_names[cls]] for cls in top_class]

   
def main():
    """
    Main function to run the prediction process
    .............
    """
    # Code for calling Predictor class and making predictions
    in_args = get_input_args()
    image_path = in_args.image_path
    check_point = in_args.check_point
    top_k = in_args.top_k
    category_names = in_args.category_names
    gpu = in_args.gpu

    predector = Predictor(image_path, check_point, top_k, category_names, gpu)

    probs, classes = predector.predict()

    for prob, cls in zip(probs, classes):
        print('*'*81)
        print('species: {}, predection probability: {}.'.format(cls, prob))


if __name__ =='__main__':
    main()