"""Text to Speech client"""

# pylint: disable=import-outside-toplevel
from logging import Logger
from pathlib import Path
from typing import Literal, TYPE_CHECKING, TypeAlias
from contextlib import suppress
from uuid import UUID, uuid4
import torchaudio
from .api import TextToSpeech
from .utils.audio import load_audio

if TYPE_CHECKING:  # pragma: no cover
    with suppress(ImportError, ModuleNotFoundError):
        from torch import Tensor


TTSMode: TypeAlias = Literal["fast", "ultra_fast", "standard"]


class TTS:
    """Text to Speech client"""

    _engine: "TextToSpeech"
    _main_audio_folder: Path
    _output_folder: Path
    _models_folder: Path
    _log: Logger

    def __init__(
        self,
        *,
        log: Logger,
        models_folder: Path | str | None = None,
        main_audio_folder: Path | str | None = None,
        output_folder: Path | str | None = None,
        use_deepspeed: bool = False,
        kv_cache: bool = False,
        half: bool = False,
    ) -> None:
        if models_folder is None:
            models_folder = Path.cwd().joinpath("tortoise_models")
        if main_audio_folder is None:
            main_audio_folder = Path(__file__).parent.joinpath("voices")
        if output_folder is None:
            output_folder = Path.cwd().joinpath("results")
        if isinstance(models_folder, str):
            models_folder = Path(models_folder)
        if isinstance(main_audio_folder, str):
            main_audio_folder = Path(main_audio_folder)
        if isinstance(output_folder, str):
            output_folder = Path(output_folder)
        self._main_audio_folder = main_audio_folder
        self._output_folder = output_folder
        self._models_folder = models_folder
        if not self._output_folder.exists():
            self._output_folder.mkdir(parents=True)
        if not self._models_folder.exists():
            self._models_folder.mkdir(parents=True)
        self._log = log
        self._engine = TextToSpeech(
            use_deepspeed=use_deepspeed,
            kv_cache=kv_cache,
            half=half,
            models_dir=self._models_folder.as_posix(),
        )

    def _get_voice_tensor(self, voice: str) -> "list[Tensor]":
        out: "list[Tensor]" = [
            load_audio(p.as_posix(), 22050)
            for p in self._main_audio_folder.joinpath(voice).glob("*.wav")
        ]
        return out

    def tts(  # noqa: PLR0913
        self,
        text: str,
        *,
        voice: str,
        mode: TTSMode = "fast",
        number: int = 3,
        file_id: UUID | None = None,
    ) -> list[Path]:
        """
        Converts text to speech and returns a list of paths to the generated audio files.

        Args:
        - text (str): The text to be converted to speech.
        - voice (Voices): The voice to use for the conversion.
        - mode (Mode): The mode to use for the conversion. Defaults to "fast".
        - number (int): The number of audio files to generate. Defaults to 3.

        Returns:
        - list[Path]: A list of paths to the generated audio files.
        """
        if file_id is None:
            file_id = uuid4()
        number = max(1, min(10, number))
        out: list[Path] = []
        audios = self._engine.tts_with_preset(
            text,
            voice_samples=self._get_voice_tensor(voice),
            preset=mode,
            k=number,
        )
        if number == 1:
            audios = [audios]
        for idx, audio in enumerate(audios):
            try:
                curr_filename = self._output_folder.joinpath(f"{file_id}_{idx}.wav")
                torchaudio.save(
                    curr_filename.as_posix(),
                    audio[0],
                    22050,
                )
                out.append(curr_filename)
                self._log.debug(f"Saved audio file: {curr_filename}")
            except Exception as err:
                self._log.error(f"Failed to save audio file: {err}")
        return out
