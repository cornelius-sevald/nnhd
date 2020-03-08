module Data.MNIST
  ( loadData
  )
where

import           System.IO
import qualified Data.ByteString.Lazy          as B
import           Codec.Compression.GZip
import           Data.IDX
import           Control.Error.Util
import           Control.Monad.Trans.Maybe
import           Control.Monad.Trans.Class
import qualified Data.Vector.Unboxed           as V
import           Numeric.LinearAlgebra

import           Neural.Training                ( TrainingExample(..)
                                                , TestingExample(..)
                                                )

trainingImagesPath = "./data/train-images-idx3-ubyte.gz" :: FilePath
testingImagesPath = "./data/t10k-images-idx3-ubyte.gz" :: FilePath
trainingLabelsPath = "./data/train-labels-idx1-ubyte.gz" :: FilePath
testingLabelsPath = "./data/t10k-labels-idx1-ubyte.gz" :: FilePath


loadData :: MaybeT IO ([TrainingExample Double], [TestingExample Double])
loadData = do
  -- Uncompress the data.
  lift $ putStrLn "Uncompressing files..."
  trainingImagesData <- lift $ readCompressedFile trainingImagesPath
  testingImagesData  <- lift $ readCompressedFile testingImagesPath
  trainingLabelsData <- lift $ readCompressedFile trainingLabelsPath
  testingLabelsData  <- lift $ readCompressedFile testingLabelsPath
  -- Decode IDX format.
  lift $ putStrLn "Decoding data..."
  trainingImages <- hoistMaybe $ decodeIDX trainingImagesData
  testingImages  <- hoistMaybe $ decodeIDX testingImagesData
  trainingLabels <- hoistMaybe $ decodeIDXLabels trainingLabelsData
  testingLabels  <- hoistMaybe $ decodeIDXLabels testingLabelsData
  -- Pair images and label.
  lift $ putStrLn "Pairing images and labels..."
  labeledTrainingImages <- hoistMaybe
    $ labeledDoubleData trainingLabels trainingImages
  labeledTestingImages <- hoistMaybe
    $ labeledDoubleData testingLabels testingImages
  -- Convert to a proper type.
  lift $ putStrLn "Formatting data..."
  let trainingExamples =
        map labeledImageToTrainingExample labeledTrainingImages
  let testingExamples = map labeledImageToTestingExample labeledTestingImages

  lift $ putStrLn "Done converting MNIST data."
  return (trainingExamples, testingExamples)

labeledImageToTrainingExample
  :: (Int, V.Vector Double) -> TrainingExample Double
labeledImageToTrainingExample (label, image) = TrainingExample iv lv
 where
  lv = fromList $ replicate label 0 ++ [1] ++ replicate (10 - label - 1) 0
  iv = fromList $ V.toList image

labeledImageToTestingExample :: (Int, V.Vector Double) -> TestingExample Double
labeledImageToTestingExample (label, image) = TestingExample iv label
  where iv = fromList $ V.toList image

readCompressedFile :: FilePath -> IO B.ByteString
readCompressedFile fp = decompress <$> B.readFile fp
