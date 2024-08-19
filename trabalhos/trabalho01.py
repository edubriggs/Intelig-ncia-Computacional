import tkinter as tk
import numpy as np
from tkinter import messagebox

class Hebb:
    def __init__(self, root):
        self.root = root
        self.root.title("Regra de Hebb")

        self.grid_size = 10
        self.cell_size = 30

        # Matrizes para letras A, B e entrada do usuário
        self.letra_A = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.letra_B = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.input_usuario = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Inicialização de pesos e bias
        self.weights = np.zeros((self.grid_size, self.grid_size))
        self.bias = 0
        
        # Botoes/grid
        self.create_grids()
        self.criar_buttons()

    def create_grids(self):
        self.letra_A_grid = self.create_grid(0, 0, "Letra A")
        self.letra_B_grid = self.create_grid(0, 1, "Letra B")
        self.input_usuario_grid = self.create_grid(1, 0, "Letra ?") 

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

    def criar_buttons(self):
        treinar_button = tk.Button(self.root, text="Treinar", command=self.train)
        treinar_button.grid(row=1, column=1, padx=10, pady=10)

        reconhecer_button = tk.Button(self.root, text="Adivinhar", command=self.recognize)
        reconhecer_button.grid(row=2, column=0, padx=10, pady=10)

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
        self.letra_A = self.get_matrix(self.letra_A_grid)
        self.letra_B = self.get_matrix(self.letra_B_grid)

        # Regra de aprendizagem hebbiana: atualizar pesos e vieses
        self.weights = self.letra_A - self.letra_B
        self.bias = np.sum(self.letra_A) - np.sum(self.letra_B)

    def recognize(self):
        self.input_usuario = self.get_matrix(self.input_usuario_grid)
        resposta = np.sum(self.weights * self.input_usuario) + self.bias

        if resposta > 0:
            messagebox.showinfo("Resultado", "A letra é : A")
        else:
            messagebox.showinfo("Resultado", "A letra é : B")

if __name__ == "__main__":
    root = tk.Tk()
    app = Hebb(root)
    root.mainloop()
