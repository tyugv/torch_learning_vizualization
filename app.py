import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for
import pickle

app = Flask(__name__)


def get_data(data, header, arr):
    if header in data:
        try:
            loss_item = float(data[header])
            arr.append(loss_item)

        except ValueError:
            print('get value not as number')


class Learning:

    def __init__(self, app, params=None):
        if params is None:
            self.params = {'max': [], 'mean': [], 'min': [], 'lr': [0]}
        else:
            self.params = params

    def refresh_plot(self):
        fig = plt.figure()
        plt.plot(self.params['max'])
        plt.plot(self.params['mean'])
        plt.plot(self.params['min'])
        plt.savefig('static/loss_plot.png', format='png')
        plt.close(fig)

    def refresh_params(self, data):
        get_data(data, 'mean_loss', self.params['mean'])
        get_data(data, 'min_loss', self.params['min'])
        get_data(data, 'max_loss', self.params['max'])
        get_data(data, 'lr', self.params['lr'])
        self.refresh_plot()


@app.after_request
def add_header(r):

    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        app.model
    except AttributeError:
        app.model = Learning(app)

    if request.method == 'POST':
        data = request.form.to_dict()
        app.model.refresh_params(data)

    return render_template('loss.html', url='static/loss_plot.png')


@app.route('/new_learning', methods=['POST'])
def create_new_learning():
    del app.model
    app.model = Learning(app)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
