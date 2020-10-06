import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, render_template, Response, send_file
from livereload import Server
import pickle

app = Flask(__name__)

app.params = {
    'max': [],
    'mean': [],
    'min': [],
    'lr': 0
}

def get_data(data, header, arr):

    if header in data:
        try:
            loss_item = float(data[header])
            arr.append(loss_item)

        except ValueError:
            print('get value not as number')


def refresh_plot():
    fig = plt.figure()
    plt.plot(app.params['max'])
    plt.plot(app.params['mean'])
    plt.plot(app.params['min'])
    plt.savefig('static/loss_plot.png', format='png')
    plt.close(fig)


@app.after_request
def add_header(r):

    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/', methods=['GET', 'POST'])
def loss_plotting():

    if request.method == 'POST':

        data = request.form.to_dict()

        get_data(data, 'mean_loss', app.params['mean'])
        get_data(data, 'min_loss', app.params['min'])
        get_data(data, 'max_loss', app.params['max'])
        get_data(data, 'lr', app.params['lr'])

        if len(app.params['lr']) > 0:
            # вернуть шаг обучения если был получен
            return render_template('loss.html', url='static/loss_plot.png'), 201, {'lr': app.params['lr'][-1]}

        refresh_plot()
        return send_file('static/loss_plot.png', mimetype='image/png'), 206, {'message': 'Sent'}

    return render_template('loss.html', url='static/loss_plot.png')


if __name__ == '__main__':

    server = Server(app.wsgi_app)
    server.watch('static/loss_plot.png')
    server.serve(port=5000)
