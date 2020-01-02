import random
import time

import pygame
from gtts import gTTS
from pygame import mixer


class Reward :


    possibleSentences =["Good girl hookie, this is your treat !","Hookie sit, you will have a sweet",
                       "that's my girl, you are a good bulldog"," this is your reward but don't tell mummy"]

    tooManySweet = ["No that's too many treats for you.", "you are a gluttonous dog..."]

    def __init__(self, **engine_kwargs):
        self.count = 0
        self.time = time.time()



    def playsound(self, audio_file):
        try:
            print("playing...")
            mixer.init()
            mixer.music.load(audio_file)
            mixer.music.play()
            while mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception:
            raise
        finally:
            mixer.quit()

    def sentence(self):
        index = random.randrange(0, len(self.possibleSentences))
        tts_en = gTTS(self.possibleSentences[index], 'en', slow=False)
        audio_file="welcome.mp3"
        tts_en.save(audio_file)
        self.playsound(audio_file)


if __name__ == '__main__':
    try:
        TTS = Reward()
        TTS.sentence()
    finally:
        print("[INFO] existing Reward test ...")