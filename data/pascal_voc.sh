url="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
archive_name="VOCtrainval_11-May-2012.tar"

wget "$url"
tar -xf "$archive_name"
rm "$archive_name"