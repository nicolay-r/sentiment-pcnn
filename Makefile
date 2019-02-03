install:
	# Download and install core library python dependencies
	git clone https://github.com/nicolay-r/sentiment-erc-core core
	git clone https://github.com/nicolay-r/RuSentRel data
	cd core && git checkout release_19_1
	cd data && git checkout clls-2018  # RuSentRel 1.0
	pip install -r core/dependencies.txt

download_embedding:
	wget http://rusvectores.org/static/models/rusvectores2/news_mystem_skipgram_1000_20_2015.bin.gz
	mkdir data/w2v
	mv news_mystem_skipgram_1000_20_2015.bin.gz > data/w2v/news_rusvectores2.bin.gz
