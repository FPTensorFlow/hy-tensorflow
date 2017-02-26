#! /usr/bin/env hy
(import os)
(import zipfile)
(import collections)
(import math)
(import [random [sample]])
(import [numpy :as np])
(import [tensorflow :as tf])
(import [six.moves [urllib]])

(def url "http://mattmahoney.net/dc/")

(defn maybe-download
  [filename expected-bytes]
  (setv filename (if (not (os.path.exists filename))
                   (urllib.request.urlretrieve (+ url filename) filename)
                   filename)
        statinfo (os.stat filename))
  (if (= expected-bytes statinfo.st-size)
    (print "Found and verified " filename)
    (raise (Exception "Failed to verify")))
  filename)

(def filename (maybe-download "text8.zip" 31344016))

(def words (with [f (zipfile.ZipFile filename)]
                 (-> (f.namelist)
                     first
                     (f.read)
                     tf.compat.as-str
                     .split)))

(print "Data size:" (len words))

(defn build-dict [words n]
  (setv words-trunc (-> (collections.Counter words)
                        (.most-common n))
        dictionary (dict (map (fn [[k c] i] [k (inc i)])
                              words-trunc
                              (range (len words-trunc))))
        _ (assoc dictionary "UNKNOWN" 0)
        words-idx (list (map (fn [w]
                               (if (in w dictionary)
                                 (dictionary.get w)
                                 0))
                             words)))
  (, words-idx dictionary))

(defn generate-batches [data batch-size n-skips skip-window]
  (assert (zero? (% batch-size n-skips)))
  (assert (<= n-skips (* 2 skip-window)))
  (setv n-groups (int (/ batch-size n-skips))
        span (inc (* 2 skip-window))
        ;; currently `partition` has something wrong
        ;; see https://github.com/hylang/hy/issues/1237
        ;; data-grouped (-> (cycle data)
        ;;                  (partition  span 1)
        ;;                  (partition n-groups))
        gen-pairs (fn [d]
                    (setv target (nth d skip-window)
                          d-rest (sample (+ (cut d 0 skip-window)
                                            (cut d (inc skip-window)))
                                         (* 2 skip-window)))
                    (take n-skips (zip d-rest (repeat target))))
        gen-arrays (fn [pairs]
                     (, (-> (list (map first pairs))
                            (np.array  :dtype np.int32))
                        (-> (list (map second pairs))
                            (np.array :dtype np.int32)
                            (.reshape (, batch-size 1)))))
        data-inf (cycle data)
        data-buffer (collections.deque :maxlen span)
        batch [])
  ;; fill the buffer first
  (data-buffer.extend (take span data-inf))
  (while True
    (if (= batch-size (len batch))
      (do (yield (gen-arrays batch))
          (setv batch []))
      (do (batch.extend (gen-pairs (list data-buffer)))
          ;; `first` will consume `data-inf`
          (data-buffer.append (first data-inf))))))

;; This function can be split into several small pieces
(defn run-model []
  (setv num-steps 100001
        vocabulary-size 50000
        [data-idx dictionary] (build-dict words (dec vocabulary-size))
        batch-size 128
        embedding-size 128
        skip-window 1
        num-skips 2
        valid-size 16
        valid-window 100
        valid-examples (np.random.choice valid-window valid-size :replace False)
        num-sampled 64
        ;; input data
        train-inputs (tf.placeholder tf.int32 :shape [batch-size])
        train-labels (tf.placeholder tf.int32 :shape [batch-size 1])
        valid-dataset (tf.constant valid-examples :dtype tf.int32)
        embeddings (tf.Variable (tf.random-uniform [vocabulary-size embedding-size] -1.0 -1.0))
        embed (tf.nn.embedding-lookup embeddings train-inputs)
        nce-weights (tf.Variable (tf.truncated-normal [vocabulary-size embedding-size]
                                                     :stddev (/ 1.0 (math.sqrt embedding-size))))
        nce-biases (tf.Variable (tf.zeros [vocabulary-size]))
        ;; Compute the average NCE loss for the batch.
        loss (tf.reduce-mean (tf.nn.nce-loss :weights nce-weights
                                             :biases nce-biases
                                             :labels train-labels
                                             :inputs embed
                                             :num-sampled num-sampled
                                             :num-classes vocabulary-size))
        optimizer (-> (tf.train.GradientDescentOptimizer 1.0)
                        (.minimize loss))
        norm (tf.sqrt (tf.reduce-sum (tf.square embeddings) 1 :keep-dims True))
        normalized-embeddings (/ embeddings norm)
        valid-embeddings (tf.nn.embedding-lookup normalized-embeddings valid-dataset)
        similarity (tf.matmul valid-embeddings normalized-embeddings :transpose-b True)
        init (tf.global-variables-initializer))
  (setv avg-loss 0
        data-batches (generate-batches data-idx batch-size num-skips skip-window))
  (with [sess (tf.Session)]
        (sess.run init)
        (for [step (range num-steps)]
          (setv [batch-inputs batch-labels] (next data-batches)
                feed-dict {train-inputs batch-inputs train-labels batch-labels}
                [_ loss-val] (sess.run [optimizer loss] :feed-dict feed-dict)
                avg-loss (+ avg-loss loss-val))
          (when (zero? (% step 2000))
            (do
             (print "loss:" (/ avg-loss 2000))
             (setv avg-loss 0))))))

(run-model)
