i=0
for file in /home/llr/cms/sarkar/DataHtoInv/0000/*.root
do
    echo $file
       python matching_new.py $file $i
       echo $i  
    echo blah
    i=$((i=i+1))
done
mv jetalgo_*.hdf5 /home/llr/cms/sarkar/HDF5/
