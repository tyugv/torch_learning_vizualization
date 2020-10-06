import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, render_template, Response, send_file
from livereload import Server


app = Flask(__name__)
app.debug = True
app.mean_loss_change = [1,2]

global max_loss_change
max_loss_change = []
#global mean_loss_change
#mean_loss_change = [1,2]
global min_loss_change
min_loss_change = []
global learning_rate
learning_rate = []


def get_data(data, header, arr):

    if header in data:
        try:
            loss_item = float(data[header])
            arr.append(loss_item)

        except ValueError:
            print('get value not as number')


def refresh_plot():
    fig = plt.figure()
    plt.plot(max_loss_change)
    plt.plot(app.mean_loss_change)
    plt.plot(min_loss_change)
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
        app.mean_loss_change.append(1)
        print(app.mean_loss_change, flush=True)
        get_data(data, 'mean_loss', mean_loss_change)
        get_data(data, 'min_loss', min_loss_change)
        get_data(data, 'max_loss', max_loss_change)
        get_data(data, 'lr', learning_rate)

        if len(learning_rate) > 0:
            # вернуть шаг обучения если был получен
            return render_template('loss.html', url='static/loss_plot.png'), 201, {'lr': learning_rate[-1]}

        refresh_plot()

        return send_file('static/loss_plot.png', mimetype='image/png'), 206, {'message': 'Sent'}

    #return render_template('loss.html', url='static/loss_plot.png')
    print(app.mean_loss_change, flush=True)
    return jsonify(str(app.mean_loss_change))


if __name__ == '__main__':

    server = Server(app.wsgi_app)
    server.watch('static/loss_plot.png')
    server.serve(port=5000)
