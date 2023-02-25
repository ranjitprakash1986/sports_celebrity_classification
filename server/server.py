from flask import Flask, request, jsonify
import util  # python custom script written for classification of user image

app = Flask(__name__)


@app.route("/classify_image", methods=["GET", "POST"])
def classify_image():
    """function on app server to classify the image"""
    image_data = request.form["image_data"]  # image_data in b64 format
    response = jsonify(
        util.classify_image(image_data)
    )  # jsonify the output and save in result

    response.headers.add("Access-Control-Allow-Origin", "*")

    return response


if __name__ == "__main__":
    print("Loading artifacts for Classification")
    util.load_artifacts()
    app.run(port=5000)
