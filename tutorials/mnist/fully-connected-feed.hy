#! /usr/bin/env hy
(require [hy.extra.anaphoric [*]])
(import argparse)
(import os)
(import [tensorflow :as tf])
(import [tensorflow.examples.tutorials.mnist [input-data mnist]])

(defn fill-feed-dict [data-set images-pl labels-pl batch-size fake-data]
  (setv [images-feed labels-feed] (data-set.next-batch batch-size fake-data))
  {images-pl images-feed labels-pl labels-feed})

(defn do-eval [sess eval-correct images-placeholder labels-placeholder data-set batch-size fake-data]
  (setv steps-per-epoch (/ data-set.num-examples batch-size)
        n-examples (* batch-size steps-per-epoch)
        feed-dict (fill-feed-dict data-set images-placeholder labels-placeholder batch-size fake-data)
        true-count (reduce + (repeat (sess.run eval-correct :feed-dict feed-dict) (int steps-per-epoch))))
  (print "n examples " n-examples " n correct: " true-count " precision: " (/ true-count n-examples)))

(defn run-training [flags]
  (setv data-sets (input-data.read-data-sets flags.input-data-dir flags.fake-data))
  (with [(-> tf .Graph .as-default)]
        (setv images-placeholder (tf.placeholder tf.float32 :shape (, flags.batch-size mnist.IMAGE-PIXELS))
              labels-placeholder (tf.placeholder tf.int32 :shape (, flags.batch-size))
              logits (mnist.inference images-placeholder flags.hidden1 flags.hidden2)
              loss (mnist.loss logits labels-placeholder)
              train-op (mnist.training loss flags.learning-rate)
              eval-correct (mnist.evaluation logits  labels-placeholder)
              summary (tf.summary.merge-all)
              init (tf.global-variables-initializer)
              saver (tf.train.Saver)
              sess (tf.Session)
              summary-writer (tf.summary.FileWriter flags.log-dir sess.graph))
        (sess.run init)
        (ap-each (range flags.max-steps)
                 (setv feed-dict (fill-feed-dict data-sets.train images-placeholder labels-placeholder flags.batch-size flags.fake-data)
                       [_ loss-value] (sess.run [train-op loss] :feed-dict feed-dict))
                 (when (zero? (% it 100))
                   (do (print "step:" it " loss:" loss-value)
                       (summary-writer.add-summary (sess.run summary feed-dict) it)
                       (summary-writer.flush)))
                 (when (or (zero? (% (inc it) 1000)) (= (inc it) flags.max-steps))
                   (do (saver.save sess (os.path.join flags.log-dir "model.ckpt") :global-step it)
                       (print "Validation eval:")
                       (do-eval sess eval-correct images-placeholder labels-placeholder data-sets.validation flags.batch-size flags.fake-data)
                       (print "Test eval:")
                       (do-eval sess eval-correct images-placeholder labels-placeholder data-sets.test flags.batch-size flags.fake-data))))))

(defmain [&rest args]
  (setv parser (.ArgumentParser argparse))
  (list (map (fn [[name params]] (apply parser.add-argument name params))
             [[["--learning-rate"] {"type" float "default" 0.01 "help" "Initial learning rate."}]
              [["--max-steps"] {"type" int "default" 2000 "help" "Number of steps to run trainer."}]
              [["--hidden1"] {"type" int "default" 128 "help" "Number of units in hidden layer 1."}]
              [["--hidden2"] {"type" int "default" 32 "help" "Number of units in hidden layer 2."}]
              [["--batch-size"] {"type" int "default" 100 "help" "Batch size.  Must divide evenly into the dataset sizes."}]
              [["--input-data-dir"] {"type" str "default" "/tmp/tensorflow/mnist/input_data" "help" "Directory to put the input data."}]
              [["--log-dir"] {"type" str "default" "/tmp/tensorflow/mnist/logs/fully_connected_feed" "help" "Directory to put the log data."}]
              [["--fake-data"] {"default" False "help" "If true, uses fake data for unit testing." "action" "store_true"}]]))
  (setv [flags unparsed] (.parse-known-args parser))
  (run-training flags))
