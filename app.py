from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from doc_copy import classify
from doc_copy_copy import classify_img
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = "supersecretkey"
app.config["UPLOAD_FOLDER"] = r"G:\icosmic_submission"


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


@app.route('/', methods=["GET", "POST"])
@app.route("/index", methods=["GET", "POST"])
def index():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        file_ext = os.path.splitext(filename)[1]
        if file_ext == '.pdf':
            file.save(os.path.join(os.path.abspath(
                os.path.dirname(__file__)), app.config["UPLOAD_FOLDER"], secure_filename("testdocument.pdf")))
            classify()
            return render_template('base.html', form=form)
        else:
            file.save(os.path.join(os.path.abspath(
                os.path.dirname(__file__)), app.config["UPLOAD_FOLDER"], secure_filename("testdocument.png")))
            classify_img()
            return render_template('base.html', form=form)

    return render_template('index.html', form=form)


if __name__ == "__main__":
    app.run(debug=True)
