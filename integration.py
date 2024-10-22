import os


def run_presentation_project():
    os.system("python3 /Users/madhandhanasekaran/Documents/PROJ/sign-language-detector-python-master/presentation/main.py")


def run_mouse_project():
    os.system("python3 /Users/madhandhanasekaran/Documents/PROJ/sign-language-detector-python-master/mouse/mouse.py")


def run_volume_project():
    os.system("python3 /Users/madhandhanasekaran/Documents/PROJ/sign-language-detector-python-master/volume/volume.py")


def main():
    while True:
        print("Please select a project to run:")
        print("1. Presentation")
        print("2. Mouse")
        print("3. Volume")
        print("4. Exit")

        choice = input("Enter your choice (1/2/3/4): ")

        if choice == '1':
            run_presentation_project()
        elif choice == '2':
            run_mouse_project()
        elif choice == '3':
            run_volume_project()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
