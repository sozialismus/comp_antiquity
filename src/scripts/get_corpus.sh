# Fetches all corpora from their source and puts them in the dat directory
echo "Creating directories"
cd /work/comp_antiquity/
mkdir -p dat
mkdir -p dat/greek
mkdir -p dat/greek/raw_data

cd dat/greek/raw_data/
echo "Fetching raw corpora - legacy data and other sources"

echo " - Cloning legacy data - Attalus, Perseus, & First1k Greek"
git clone "https://github.com/sozialismus/legacy-raw"
echo " - Cloning Online Critical Pseudepigrapha"
git clone "https://github.com/OnlineCriticalPseudepigrapha/Online-Critical-Pseudepigrapha.git"
echo " - Cloning SBL Greek New Testament"
git clone "https://github.com/LogosBible/SBLGNT"
echo " - LXX-Swete-1930 by Eliranwong"
git clone "https://github.com/eliranwong/LXX-Swete-1930.git"

echo "Moving files from legacy_raw to dat"
cd legacy-raw/
cp -r First1KGreek attalus canonical-greekLit ..

cd ..

# sepa files are not included in this run
# echo "Trying to unzip Septuagint data"
# sudo apt install unzip
# unzip "SEPA.zip" -d "raw_data/SEPA"
