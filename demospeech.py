import pyttsx3

def test_tts():
    engine = pyttsx3.init()
    engine.say("Hello, this is a test of the text to speech system.")
    engine.runAndWait()

if __name__ == "__main__":
    test_tts()
