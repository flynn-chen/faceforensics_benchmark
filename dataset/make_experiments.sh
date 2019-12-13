for m in "Xception" "ResNet152" "InceptionV3" "InceptionResNetV2";
do
	echo $m
	new_name=$(echo run_VGG_short.sh | sed "s/VGG/${m}/g")
        cp run_VGG_short.sh ${new_name}
        sed -i "s/VGG/${m}/g" ${new_name}
        sbatch ${new_name}
done
echo "VGG"
sbatch run_VGG_short.sh


for m in "Xception" "ResNet152" "InceptionV3" "InceptionResNetV2";
do
	echo $m
	new_name=$(echo run_VGG.sh | sed "s/VGG/${m}/g")
	cp run_VGG.sh ${new_name}
	sed -i "s/VGG/${m}/g" ${new_name}
	sbatch ${new_name}
done
echo "VGG"
sbatch run_VGG.sh
