import tkinter as tk
import numpy as np
from tkinter import messagebox

class Perceptron:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptron - Letra A ou B")

        self.grid_size = 10
        self.cell_size = 30

        # Matrizes para letras A, B e entrada do usuário
        self.letter_A = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.letter_B = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.user_input = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Inicialização de pesos e bias
        self.weights = np.zeros((self.grid_size, self.grid_size))
        self.bias = 0
        self.learning_rate = 0.1  # Taxa de aprendizagem

        self.create_grids()

        # Botões
        self.create_buttons()

    def create_grids(self):
        self.letter_A_grid = self.create_grid(0, 0, "Letra A")
        self.letter_B_grid = self.create_grid(0, 1, "Letra B")
        self.user_input_grid = self.create_grid(1, 0, "Letra ?")

    def create_grid(self, row, col, label_text):
        frame = tk.Frame(self.root)
        frame.grid(row=row, column=col, padx=20, pady=20)

        label = tk.Label(frame, text=label_text)
        label.grid(row=0, column=0, columnspan=self.grid_size)

        grid = []
        for i in range(self.grid_size):
            row_cells = []
            for j in range(self.grid_size):
                cell = tk.Button(frame, width=2, height=1, bg="white",
                                 command=lambda i=i, j=j, g=grid: self.toggle_cell(i, j, g))
                cell.grid(row=i+1, column=j)

                row_cells.append(cell)
            grid.append(row_cells)
        return grid

    def create_buttons(self):
        train_button = tk.Button(self.root, text="Treinar", command=self.train)
        train_button.grid(row=1, column=1, padx=10, pady=10)

        recognize_button = tk.Button(self.root, text="Adivinhar", command=self.recognize)
        recognize_button.grid(row=2, column=0, padx=10, pady=10)

    def toggle_cell(self, i, j, grid):
        cell = grid[i][j]
        if cell["bg"] == "white":
            cell.config(bg="black")
        else:
            cell.config(bg="white")

    def get_matrix(self, grid):
        matrix = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i][j]["bg"] == "black":
                    matrix[i][j] = 1
        return matrix

    def train(self):
        self.letter_A = self.get_matrix(self.letter_A_grid)
        self.letter_B = self.get_matrix(self.letter_B_grid)

        # Treinamento para A (saída desejada: 1) e B (saída desejada: -1)
        self.perceptron_train(self.letter_A, 1)
        self.perceptron_train(self.letter_B, -1)

    def perceptron_train(self, input_matrix, target):
        output = np.sum(self.weights * input_matrix) + self.bias
        prediction = 1 if output > 0 else -1
        error = target - prediction

        # Atualizando pesos e bias com base no erro
        self.weights += self.learning_rate * error * input_matrix
        self.bias += self.learning_rate * error

    def recognize(self):
        self.user_input = self.get_matrix(self.user_input_grid)
        response = np.sum(self.weights * self.user_input) + self.bias
        prediction = 1 if response > 0 else -1

        if prediction == 1:
            messagebox.showinfo("Resultado", "A letra é: A")
        else:
            messagebox.showinfo("Resultado", "A letra é: B")

if __name__ == "__main__":
    root = tk.Tk()
    app = Perceptron(root)
    root.mainloop()
