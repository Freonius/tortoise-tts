from logging import getLogger
from tortoise.client import TTS

log = getLogger("tortoise")
tts = TTS(log=log)


tts.tts("Hello, how are you?", voice="emma", mode="fast", number=1)
