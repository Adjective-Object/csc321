import sys

def analyzecsv(fpath):
	count = 0
	personMatches = 0
	genderMatches = 0
	for line in open(fpath).readlines():
		line = [s.strip() for s in line.split(",")]
		if int(line[-2]):
			personMatches += 1

		if int(line[-1]):
			genderMatches += 1

		count += 1

	return (float(genderMatches) / count,
			float(personMatches) / count)

def printHumanResults(path, r):
	sys.stdout.write("%s: " % path)
	print ("gender accuracy: %2.2f%%, name accuracy: %2.2f%%"
			% tuple(100 * s for s in r))

def main():
	results = [analyzecsv(fpath) for fpath in sys.argv[1:]]
	pathRes = list(zip(sys.argv[1:], results))

	for path, r in pathRes:
		printHumanResults(path, r)

	bestByGender = max(pathRes, 
		key=lambda (path, (gender, person)): gender)
	bestByPerson = max(pathRes, 
		key=lambda (path, (gender, person)): person)

	print
	print "best set by gender: %s  (%2.4f %%)" % (
		bestByGender[0], bestByGender[1][0] * 100)
	print "best set by Person: %s  (%2.4f %%)" % (
		bestByPerson[0], bestByPerson[1][1] * 100)




if __name__ == "__main__":
	main()