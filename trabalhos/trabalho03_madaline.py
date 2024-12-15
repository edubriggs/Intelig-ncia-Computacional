import numpy as np
import tkinter as tk
from tkinter import messagebox

class Madaline:
    def __init__(self, root):
        self.root = root
        self.root.title("Madaline - Reconhecimento de Letras")

        self.grid_size = 10
        self.cell_size = 30

        self.letras = ['A', 'B', 'C', 'D', 'E']
        self.input_usuario = np.zeros((self.grid_size, self.grid_size), dtype=int)

        self.hidden_weights = np.random.rand(self.grid_size * self.grid_size, 5) - 0.5
        self.output_weights = np.random.rand(5, len(self.letras)) - 0.5
        self.hidden_bias = np.random.rand(5) - 0.5
        self.output_bias = np.random.rand(len(self.letras)) - 0.5

        self.create_grids()
        self.create_buttons()

    def create_grids(self):
        self.grids = {}
        for i, letter in enumerate(self.letras):
            self.grids[letter] = self.create_grid(i // 3, i % 3, f"Letra {letter}")

        self.input_usuario_grid = self.create_grid(2, 1, "Entrada do Usuário")

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
                cell.grid(row=i + 1, column=j)
                row_cells.append(cell)
            grid.append(row_cells)
        return grid

    def create_buttons(self):
        treinar_button = tk.Button(self.root, text="Treinar", command=self.train)
        treinar_button.grid(row=3, column=0, padx=10, pady=10)

        reconhecer_button = tk.Button(self.root, text="Reconhecer", command=self.recognize)
        reconhecer_button.grid(row=3, column=2, padx=10, pady=10)

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
        inputs = []
        targets = []

        for i, letter in enumerate(self.letras):
            matrix = self.get_matrix(self.grids[letter]).flatten()
            inputs.append(matrix)
            target = np.zeros(len(self.letras))
            target[i] = 1
            targets.append(target)

        inputs = np.array(inputs)
        targets = np.array(targets)

        learning_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            total_error = 0
            for x, t in zip(inputs, targets):
                hidden_input = np.dot(x, self.hidden_weights) + self.hidden_bias
                hidden_output = self.sigmoid(hidden_input)
                final_input = np.dot(hidden_output, self.output_weights) + self.output_bias
                final_output = self.sigmoid(final_input)

                output_error = t - final_output
                hidden_error = np.dot(self.output_weights, output_error) * hidden_output * (1 - hidden_output)

                self.output_weights += learning_rate * np.outer(hidden_output, output_error)
                self.output_bias += learning_rate * output_error
                self.hidden_weights += learning_rate * np.outer(x, hidden_error)
                self.hidden_bias += learning_rate * hidden_error

                total_error += np.sum(output_error ** 2)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Erro Total: {total_error}")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def recognize(self):
        input_matrix = self.get_matrix(self.input_usuario_grid).flatten()

        hidden_input = np.dot(input_matrix, self.hidden_weights) + self.hidden_bias
        hidden_output = self.sigmoid(hidden_input)
        final_input = np.dot(hidden_output, self.output_weights) + self.output_bias
        final_output = self.sigmoid(final_input)

        predicted_index = np.argmax(final_output)
        confidence = final_output[predicted_index]

        if confidence >= 0.5:
            predicted_letter = self.letras[predicted_index]
            messagebox.showinfo("Resultado", f"A letra é: {predicted_letter} com confiança {confidence:.2f}")
        else:
            messagebox.showwarning("Resultado", "A letra não foi reconhecida com confiança suficiente.")

if __name__ == "__main__":
    root = tk.Tk()
    app = Madaline(root)
    root.mainloop()
