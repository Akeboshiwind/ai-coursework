(ns ai-coursework.train
  (:require [ai-coursework.utils :as u]
            [ai-coursework.network :as n]))

(defn run
  "Make a forward pass through the network with the given input to generate a
  map of outputs for each neuron."
  [network input]
  (assert (= (:width (first (:layers network)))
             (count input))
          "The input must be the same size as the first layer.")
  (let [input-values (zipmap (:neurons (first (:layers network)))
                             input)]
    (reduce (fn [outputs {:keys [neurons activation-fn] :as layer}]
              (let [activation-fn (:fn activation-fn)
                    new-sums (zipmap neurons
                                     (map (fn [n-id]
                                            (let [neuron (n/get-neuron network n-id)
                                                  parents (n/get-parents network n-id)
                                                  parent-outputs (map #(get outputs %) parents)]
                                              (apply +
                                                     (:bias neuron)
                                                     (map * (:weights neuron) parent-outputs))))
                                          neurons))
                    new-outputs (u/fmap activation-fn new-sums)]
                (merge outputs new-outputs)))
            input-values
            (rest (:layers network)))))

(defn run-with-output
  [network input]
  (let [outputs (run network input)
        output-layer (last (:layers network))
        output-neurons (:neurons output-layer)]
    (map outputs output-neurons)))

(binding [*unchecked-math* :warn-on-boxed]
  (defn learn
    "Output a new network with adjusted weights and biases that should work
  slightly better than before."
    [network {:keys [input expected] :as example}]
    (let [outputs (run network input)
          learning-rate (:learning-rate network)
          output-layer (last (:layers network))
          output-neurons (:neurons output-layer)
          out-neuron->expected (zipmap output-neurons
                                       expected)
          new-network (doall (reduce (fn [network layer]
                                       (let [{:keys [deltas neurons]}
                                             (doall (reduce (fn [{:keys [deltas neurons] :as out} n-id]
                                                              (let [neuron (n/get-neuron network n-id)
                                                                    output (get outputs n-id)
                                                                    deriv (:deriv (:activation-fn output-layer))
                                                                    children (n/get-children network n-id)
                                                                    child-deltas (map #(get deltas %) children)
                                                                    deriv-output (deriv output)
                                                                    neuron-delta (if (nil? children)
                                                                                   (let [expected-value (get out-neuron->expected
                                                                                                             n-id)]
                                                                                     (* deriv-output
                                                                                        (- expected-value output)))
                                                                                   (let [parent-ordering (n/get-parents network (first children))
                                                                                         idx (.indexOf parent-ordering n-id)
                                                                                         neuron->child-weights (map (fn [c-id]
                                                                                                                      (-> (n/get-neuron network c-id)
                                                                                                                          (:weights)
                                                                                                                          (nth idx)))
                                                                                                                    children)]
                                                                                     (* deriv-output
                                                                                        (apply +
                                                                                               (map *
                                                                                                    child-deltas
                                                                                                    neuron->child-weights)))))
                                                                    parents (n/get-parents network n-id)
                                                                    new-weights (doall (map (fn [[p-id weight]]
                                                                                              (let [parent (n/get-neuron network p-id)
                                                                                                    parent-output (get outputs p-id)]
                                                                                                (+ weight
                                                                                                   (* learning-rate
                                                                                                      neuron-delta
                                                                                                      parent-output))))
                                                                                            (map vector parents (:weights neuron))))
                                                                    new-bias (+ (:bias neuron)
                                                                                (* learning-rate
                                                                                   neuron-delta))]
                                                                {:deltas (assoc! deltas n-id neuron-delta)
                                                                 :neurons (let [n (get neurons n-id)
                                                                                new-n (assoc n :bias new-bias
                                                                                             :weights new-weights)]
                                                                            (assoc! neurons n-id new-n))}))
                                                            {:deltas (:deltas network)
                                                             :neurons (:neurons network)}
                                                            (:neurons layer)))]
                                         (-> network
                                             (assoc :deltas deltas)
                                             (assoc :neurons neurons))))
                                     (-> network
                                         (assoc :neurons (transient (:neurons network)))
                                         (assoc :deltas (transient {})))
                                     (reverse (rest (:layers network)))))]
      (-> new-network
          (assoc :neurons (persistent! (:neurons new-network)))
          (assoc :deltas (persistent! (:deltas new-network)))))))

(defn train
  "Iterate over the list of examples calling learn on each one to improve the
  network until the stop-fn is satisfied."
  [network examples stop-fn]
  (loop [network (assoc network :epoch 1)]
    (let [new-network (reduce learn network examples)]
      (if (stop-fn network)
        network
        (recur (update new-network :epoch inc))))))

(defmacro combine
  "Return a function that checks that it's inputs return true when applied to all fns.
  Does short circuit."
  [& fns]
  (let [n (gensym "n")]
    `(fn [~n]
       (or ~@(map (fn [fn]
                    `(~fn ~n))
                  fns)))))
(defn max-epoch
  "Return a function that checks the epoch of the network is over max."
  [max]
  (fn [network]
    (when (>= (:epoch network)
              max)
      network)))

(defn check-n-epochs
  "Return a function that returns true when it has been n epochs."
  [n]
  (fn [network]
    (when (= 0 (mod (:epoch network) n))
      (println "Epoch: " (:epoch network)))))

(defn mean-squared-error
  "Calculate the mean squared error"
  [actual expected]
  (/ (apply + (map (fn [a e]
                     (Math/pow (- a e) 2))
                   actual
                   expected))
     (count actual)))

(defn network-mean-error
  [network examples]
  (let [[sum count] (reduce (fn [[sum count] {:keys [input expected]}]
                              (let [actual (run-with-output network input)]
                                [(+ sum (mean-squared-error actual expected))
                                 (+ 1 count)]))
                            [0 0]
                            examples)]
    (/ sum count)))

(defn early-stopping
  [test-set epoch-check]
  (assert (not= 0 (count test-set))
          "Validation set must contain some examples.")
  (let [state (atom {:last-error nil :last-network nil})]
    (fn [network]
      (when (= 0 (mod (:epoch network) epoch-check))
        (let [mean-error (network-mean-error network test-set)
              last-error (:last-error @state)]
          (if (and (not (nil? last-error))
                   (> mean-error last-error))
            network
            (do
              (reset! state {:last-error network
                             :last-network network})
              false)))))))

(def last-network (atom nil))
(defn save-network
  [network]
  (reset! last-network network)
  false)
