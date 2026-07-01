# sea-animal-detection-system
detecting sea creatures [Jellyfish, Harbor seals, Penguins, Fish, Dolphins, and Sharks]


## The Algorithm

I first create a file full of the labels of the animals I'm going to use to not forget.

When I finished, I ran the docker container and went to my detection ssd folder: 
          cd ~/jetson-inference/
           ./docker/run.sh

           cd python/training/detection/ssd

Then, I downloaded all the photos I needed to train the ai to know what animal is which. I chose to download 5000 photos for this:

          python3 open_images_downloader.py --max-images=5000 \
          --class-names "Jellyfish,Harbor seal,Fish,Dolphin,Penguin,Shark" \
          --data=data/seaanimals

After this, I run a code which makes the ai look over the photos 30 times just to make sure it knows what animal is what:

           python3 train_ssd.py --data=data/seaanimals --model-dir=models/seaanimals --batch-size=4 --epochs=30

Now before scanning any video, I have to run this code to make it possible to actually do it:

           python3 onnx_export.py --model-dir=models/seaanimals

Lastly, we have to download a video and make the ai scan it for any animal is recognizes, I used this code for it. Depending on the name of the file of the video and what name you want the output of the video to be, you have to change some things (That are mainly naming related):

           detectnet --model=models/seaanimals/ssd-mobilenet.onnx --labels=models/seaanimals/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes  data/VIDEONAME.mp4 data/output.mp4
I have added two videos as demonstration in the demo folder, if you click on them, you should see the ai I have trained to detect the sea animals.

