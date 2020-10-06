import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, render_template, Response, send_file
from livereload import Server
import pickle


def write_to_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    f.close()


def read_from_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data


app = Flask(__name__)

app.param = {
    'max': [],
    'mean': [],
    'min': [],
    'lr': 0
}
write_to_pickle('parametrs.pickle', app.param)

def get_data(data, header, arr):

    if header in data:
        try:
            loss_item = float(data[header])
            arr.append(loss_item)

        except ValueError:
            print('get value not as number')


def refresh_plot():
    fig = plt.figure()
    plt.plot(app.param['max'])
    plt.plot(app.param['mean'])
    plt.plot(app.param['min'])
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
        param = read_from_pickle('parametrs.pickle')
        print(param, flush=True)
        get_data(data, 'mean_loss', param['mean'])
        param['mean'].append(1)
        get_data(data, 'min_loss', param['min'])
        get_data(data, 'max_loss', param['max'])
        get_data(data, 'lr', param['lr'])
        write_to_pickle('parametrs.pickle', param)

        #if len(learning_rate) > 0:
        #    # вернуть шаг обучения если был получен
        #    return render_template('loss.html', url='static/loss_plot.png'), 201, {'lr': learning_rate[-1]}

        refresh_plot()

        return send_file('static/loss_plot.png', mimetype='image/png'), 206, {'message': 'Sent'}
    print(read_from_pickle('parametrs.pickle'), flush=True)
    return render_template('loss.html', url='static/loss_plot.png')


if __name__ == '__main__':

    server = Server(app.wsgi_app)
    server.watch('static/loss_plot.png')
    server.serve(port=5000)
