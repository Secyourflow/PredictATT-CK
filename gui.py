# Version:    1.00
# Author:     Aryan Shrestha
# Interface:  Graphical
# Date:       2024_8_20

from flask import Flask, render_template, request, send_file
import joblib
import re

import classification_tools.preprocessing as prp
import classification_tools.postprocessing as pop
import classification_tools.save_results as sr
import classification_tools as clt

from operator import itemgetter

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/save', methods=['POST'])
def save():
    """
    Save predictions either in the training set or in a JSON file under STIX format.
    """
    if request.method == 'POST':
        form_data = request.form.to_dict()
        save_to_file = "filesave" in form_data
        save_to_train = "trainsave" in form_data
        references = []

        # Save to a JSON file in STIX format
        if save_to_file:
            for key in form_data:
                if key in clt.ALL_TTPS:
                    references.append(clt.STIX_IDENTIFIERS[clt.ALL_TTPS.index(key)])
            report = re.sub("\r\n", " ", form_data['hidereport'])
            file_to_save = sr.save_results_in_file(report, form_data['name'], form_data['date'], references)
            return send_file(file_to_save, as_attachment=True)

        # Save to the custom training set
        if save_to_train:
            for key in form_data:
                if key in clt.ALL_TTPS:
                    references.append(key)
            report = re.sub("\r\n", "\t", prp.remove_u(form_data['hidereport'].encode('utf8').decode('ISO-8859-1')))
            sr.save_to_train_set(report, references)

    return ('', 204)

@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Retrain the classifier with the new data added by the user.
    """
    if request.method == 'POST':
        clt.train(False)
    return ('', 204)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the techniques and tactics for the report entered by the user.
    """
    if request.method == 'POST':
        report = prp.remove_u(request.form['message'].encode('utf8').decode('ISO-8859-1'))

        # Load post-processing parameters and min-max confidence scores
        parameters = joblib.load("classification_tools/data/configuration.joblib")
        min_prob_tactics, max_prob_tactics = parameters[2]
        min_prob_techniques, max_prob_techniques = parameters[3]

        # Predict tactics and techniques
        pred_tactics, predprob_tactics, pred_techniques, predprob_techniques = clt.predict(report, parameters)

        # Process and prepare results for display
        pred_to_display_tactics = [
            [clt.CODE_TACTICS[i], clt.NAME_TACTICS[i], pred_tactics[0][i], 
             max(0.0, min(1.0, (predprob_tactics[0][i] - min_prob_tactics) / (max_prob_tactics - min_prob_tactics))) * 100]
            for i in range(len(predprob_tactics[0]))
        ]
        pred_to_display_techniques = [
            [clt.CODE_TECHNIQUES[j], clt.NAME_TECHNIQUES[j], pred_techniques[0][j], 
             max(0.0, min(1.0, (predprob_techniques[0][j] - min_prob_techniques) / (max_prob_techniques - min_prob_techniques))) * 100]
            for j in range(len(predprob_techniques[0]))
        ]

        # Sort results by confidence score in descending order
        pred_to_display_tactics = sorted(pred_to_display_tactics, key=itemgetter(3), reverse=True)
        pred_to_display_techniques = sorted(pred_to_display_techniques, key=itemgetter(3), reverse=True)

        return render_template('result.html', report=request.form['message'], 
                               predictiontact=pred_to_display_tactics, 
                               predictiontech=pred_to_display_techniques)

if __name__ == '__main__':
    app.run(debug=True)
