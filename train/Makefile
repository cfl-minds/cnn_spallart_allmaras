RM = /bin/rm
py3 = python3
clean:
	$(RM) -f generator_at_epoch_*.h5 model_at_epoch_*.h5 Validation_trainingLosses.txt trainingLosses.txt TrainingCurve.pdf
	$(RM) -rf __pycache__ 

train:
	$(py3) train.py
