# abcm2ps install
cd ..
git clone https://github.com/lewdlime/abcm2ps.git
cd abcm2ps
./configure && make
echo "export PATH=\"$(pwd):\$PATH\"" >> ~/.bashrc

# timidity install
cd ..
git clone https://github.com/geofft/timidity.git
cd timidity
./configure && make && make install
cd timidity
echo "export PATH=\"$(pwd):\$PATH\"" >> ~/.bashrc

