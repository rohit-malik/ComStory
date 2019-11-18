use python3

Usage:
set up conda env :: conda create -n fer2013 python=3 anaconda

set up lib :: pip install -r requirements.txt

import the lib in code as  :: import fertestcustom as f

run the method which will result the predict emotion on input image as str type objct :: result = f.predict_emotion("image.jpg")
