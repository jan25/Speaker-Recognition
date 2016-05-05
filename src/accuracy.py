

def percent(inputs, target, ann):
	ok = 0
	for i in range(inputs.shape[0]):
		if check(ann.test(inputs[i]), target[i], True):
			ok += 1
	return (ok / len(inputs)) * 100

def sindex(out):
	output = 0
	for i in range(len(out)):
		if out[i] >= out[output]:
			output = i
	return output

# check if output vector and target are same
def check(out, tar, status = False):
	if len(out) != len(tar):
		raise Exception('output and target not of same length')
	output = sindex(out)
	if status:
		if tar[output] == 1:
			print ('OK: recognized as speaker', (output + 1))
		else:
			print ('ERROR: not recognized')
	return tar[output] == 1

