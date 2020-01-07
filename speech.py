import speech_recognition as sr
import pytesseract
r=sr.Recognizer()
with sr.Microphone() as source:
    print("Speak Anything")
    audio=r.listen(source)

    try:
        text=r.recognize_google(audio)
        print('You Said:{}'.format(text))
    except:
        print("Sorry could not recognize your voice")