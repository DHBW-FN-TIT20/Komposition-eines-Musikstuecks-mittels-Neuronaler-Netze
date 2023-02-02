import keras
import music21 as m21
import json
import numpy as np
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class MelodyGenerator:

    def __init__(self, model_path="model.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature ):

        # create seed with start symbol
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to integer
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            
            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))

            # make 3d array
            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed 
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check wether we are at the end of a melody
            if output_symbol == "/":
                break

            #update melody
            melody.append(output_symbol)
        
        return melody


    def _sample_with_temperature(self, probabilities, temperature):
        # temperature  -> infinity 
        # temperature -> 0
        # temperature = 1
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) /np.sum(np.exp(predictions))

        choices = range(len(probabilities)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.midi"):
        # create a music21 stream
        stream = m21.stream.Stream()

        # create melody part
        melodyPart = m21.stream.Part()
        melodyPart.insert(m21.instrument.Piano())

        secondPart = m21.stream.Part()
        secondPart.insert(m21.instrument.Piano())

        thirdPart = m21.stream.Part()
        thirdPart.insert(m21.instrument.Piano())

        # create optional base drum part
        bassPart = m21.stream.Part()
        bassDrum = m21.instrument.BassDrum()
        bassDrum.midiChannel = 3
        bassPart.insert(bassDrum)

        # create optional snare drum part
        snarePart = m21.stream.Part()
        snareDrum = m21.instrument.SnareDrum()
        snareDrum.midiChannel = 2
        snarePart.insert(snareDrum)

                      
        # parse all symbols in the melody and create note/rest objects
        # Example melody: 60 _ _ _ r _ 62 _
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we are dealing with note/rest beyond the first one
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1 -> quarter note

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                        m21_event2 = m21.note.Rest(quarterLength=quarter_length_duration)
                        m21_event3 = m21.note.Rest(quarterLength=quarter_length_duration)
                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                        m21_event2 = m21.note.Note(int(start_symbol)+4, quarterLength=quarter_length_duration)
                        m21_event3 = m21.note.Note(int(start_symbol)+7, quarterLength=quarter_length_duration)

                    melodyPart.append(m21_event)
                    secondPart.append(m21_event2)
                    thirdPart.append(m21_event3)
                    step_counter = 1

                start_symbol = symbol
                
            # handle case in which we have an prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to midi file
        stream.append(melodyPart)
        # stream.insert(0,secondPart)
        # stream.insert(0, thirdPart)


        # bass = m21.note.Note(35, quarterLength=4)
        # snare = m21.note.Note(38, quarterLength=1)
        # n = int(melodyPart.duration.quarterLength / bass.duration.quarterLength)
        # bassPart.repeatAppend(bass, n)
        # snarePart.repeatAppend(snare, int(melodyPart.duration.quarterLength))
        # stream.append(bassPart)
        # stream.append(snarePart)
        
        stream.write(format, file_name)

if __name__ == "__main__":
    mg = MelodyGenerator(model_path="model_bach.h5")
    seed = "69 _ 65 _ 67 _ 55 _ r 60 55 52 48 _ _ _"
    seed2 = "60 _ 62 _ 52 _ 50 _ 52 _ 48 _ 53 _ 52 _ 53 _ 50 _ 56 _ 52 _ 57 _ 50 _ 52 _ 50 _ 52 _ 40 _ 45 _ _ _ 43 _ _ _ 41 _ _ _ _ _ _ _ 40 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _"
    melody = mg.generate_melody(seed2, 1000, SEQUENCE_LENGTH, 2.1)
    print(melody)
    mg.save_melody(melody, file_name="reborn_bach.mid")
