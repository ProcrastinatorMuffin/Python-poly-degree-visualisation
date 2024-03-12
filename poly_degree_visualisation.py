from manim import *
import numpy as np
from sklearn.linear_model import Ridge as RR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


class PolyDegreeVisualization(Scene):
    def construct(self):
        self.setup_colors()
        self.seed_value = 1
        self.max_poly_degree = 12
        self.poly_degree_step = 2
        self.model_circle_scale = 15
        self.error_circle_scale = 10
        self.font_size = 32

        self.generate_data()
        self.setup_scene()
        self.animate_polynomial_degrees()

    def setup_colors(self):
        self.colors = {
            'lb': LIGHT_BROWN,
            'ba': BLUE_A,
            'ra': RED_A,
            'p': PURPLE_E,
            'bc': BLUE_C,
            'w': WHITE,
            'g': GOLD}

    def generate_data(self):
        np.random.seed(self.seed_value)

        def f(x):
            return x * np.sin(x)

        n_total = 100
        x_total = np.linspace(0, 10, n_total)
        y = f(x_total) + np.random.randn(n_total)
        X = x_total[:, np.newaxis]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.8, random_state=42)
        self.n_train, self.n_test = self.X_train.shape[0], self.X_test.shape[0]

    def setup_scene(self):
        self.axes_data = Axes(
            x_range=[-1, 11.2, 0.1],
            y_range=[-10, 10, 0.1],
            axis_config={"color": self.colors['w']},
            tips=False,
        )
        self.axes_display = Axes(
            x_range=[-1, 11.2, 2],
            y_range=[-10, 10, 2],
            axis_config={"color": self.colors['p']},
            tips=True,
        )

        dot_radius = 0.08

        G = VGroup()
        for i in range(self.n_train):
            point = self.axes_data.c2p(self.X_train[i, 0], self.y_train[i])
            G += Dot([point], radius=dot_radius, color=random_color())
        self.add(G)

        self.plotting_objects = {'polynomial_graphs': [], 'degree': [], 'degree_circle': [], 'training_error': [],
                                 'validation_error': [], 'training_error_circle': [], 'validation_error_circle': []}

        labels = self.axes_data.get_axis_labels(Tex("$x$", color=self.colors['p']).scale(0.7),
                                                Tex("$\hat{y}$", color=self.colors['p']).scale(0.7))
        self.add(MathTex('\hat{y_i}=f(x_i)', color=self.colors['bc']).to_edge(UR, buff=0.5))
        t = Tex('Regression Visualization', color=self.colors['w']).to_edge(UP, buff=0.5)
        box = SurroundingRectangle(t, color=self.colors['p'], buff=0.2)
        self.add(self.axes_display, t, labels, box)

    def train_model(self, X_train, y_train, degree, alpha):
        model = make_pipeline(PolynomialFeatures(degree), RR(alpha=alpha))
        model.fit(X_train, y_train)
        return model

    def make_predictions(self, x, model):
        x2 = np.array([x])
        return model.predict(x2[:, np.newaxis]).item()

    def mse(self, model):
        r = np.sum((model.predict(self.X_train) - self.y_train) ** 2) / self.n_train
        rv = np.sum((model.predict(self.X_test) - self.y_test) ** 2) / self.n_test
        return r, rv

    def animate_polynomial_degrees(self):
        def f(x):
            return x * np.sin(x)

        for d in range(1, self.max_poly_degree, self.poly_degree_step):
            model = self.train_model(self.X_train, self.y_train, d, 0)
            self.plotting_objects['polynomial_graphs'].append(
                self.axes_data.plot(lambda x: self.make_predictions(x, model), color=self.colors['lb']))
            r, rv = self.mse(model)
            self.plotting_objects['training_error'].append(
                Tex("MSE(Train) = {:.2f}".format(r), font_size=self.font_size - 10, color=self.colors['bc']).to_edge(
                    DOWN + LEFT,
                    buff=0.5).shift(
                    RIGHT))
            self.plotting_objects['validation_error'].append(
                Tex("MSE(Val) = {:.2f}".format(rv), font_size=self.font_size - 10, color=self.colors['ba']).to_edge(
                    DOWN, buff=0.5))
            self.plotting_objects['degree'].append(
                Tex(f'Polynomial degree =  {d}', font_size=self.font_size - 10, color=self.colors['ra']).to_edge(
                    DOWN + RIGHT, buff=0.5))
            self.plotting_objects['validation_error_circle'].append(
                Circle(radius=rv / self.error_circle_scale, color=self.colors['ba'], fill_opacity=0.5).next_to(
                    self.plotting_objects['validation_error'][-1], UP, buff=0.1))
            self.plotting_objects['training_error_circle'].append(
                Circle(radius=r / self.error_circle_scale, color=self.colors['bc'], fill_opacity=0.5).next_to(
                    self.plotting_objects['training_error'][-1], UP, buff=0.1))
            self.plotting_objects['degree_circle'].append(
                Circle(radius=d / self.model_circle_scale, color=self.colors['ra'], fill_opacity=0.5).next_to(
                    self.plotting_objects['degree'][-1], UP, buff=0.1))

        self.play(Write(self.plotting_objects['degree'][0]),
                  Write(self.plotting_objects['training_error'][0]),
                  Write(self.plotting_objects['validation_error'][0]),
                  Write(self.plotting_objects['polynomial_graphs'][0]),
                  Write(self.plotting_objects['degree_circle'][0]),
                  Write(self.plotting_objects['training_error_circle'][0]),
                  Write(self.plotting_objects['validation_error_circle'][0]),
                  run_time=2)

        for i in range(1, len(self.plotting_objects['polynomial_graphs'])):
            self.play(Transform(self.plotting_objects['degree'][0], self.plotting_objects['degree'][i]),
                      Transform(self.plotting_objects['training_error'][0], self.plotting_objects['training_error'][i]),
                      Transform(self.plotting_objects['validation_error'][0],
                                self.plotting_objects['validation_error'][i]),
                      Transform(self.plotting_objects['polynomial_graphs'][0],
                                self.plotting_objects['polynomial_graphs'][i]),
                      Transform(self.plotting_objects['degree_circle'][0], self.plotting_objects['degree_circle'][i]),
                      Transform(self.plotting_objects['training_error_circle'][0],
                                self.plotting_objects['training_error_circle'][i]),
                      Transform(self.plotting_objects['validation_error_circle'][0],
                                self.plotting_objects['validation_error_circle'][i]),
                      run_time=3)

        label_gt = Tex('Ground Truth', color=self.colors['w']).to_edge(UP).shift(2 * DOWN)
        self.play(Write(self.axes_data.plot(lambda x: f(x), color=self.colors['w'], stroke_width=4)), Write(label_gt),
                  run_time=3)
        self.wait(3)


if __name__ == "__main__":
    scene = PolyDegreeVisualization()
    scene.render()
