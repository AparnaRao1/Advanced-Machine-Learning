from data import generate_xor_data
from visual import plot_2d_data

def main():
    X,y = generate_xor_data()
    plot_2d_data(X,y,title = "Non linear Data/ linearly unseperable/xor")

if __name__ == "__main__":
    main()