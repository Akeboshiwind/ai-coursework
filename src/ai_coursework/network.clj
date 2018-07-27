(ns ai-coursework.network
  (:require [ai-coursework.utils :as u]))

(defn find-children
  "Inverts the map of child->[parent] into a map of parent->[child]"
  [parents]
  (u/invert-map parents))


(defn build
  "Builds a neural network from a network description."
  [desc learning-rate & opts]
  (let [network (reduce (fn [{:keys [layers neurons parents]} layer]
                          (let [last-layer (last layers)
                                parent-count (if (nil? last-layer)
                                               1; if this is the first layer, assume it's the input layer
                                               (count (:neurons last-layer)))
                                new-neurons (reduce (fn [neurons id]
                                                      (assoc neurons id
                                        ; create new neuron
                                                             {:id id
                                                              :bias ((:bias-fn layer))
                                                              :weights (repeatedly parent-count (:weight-fn layer))}))
                                                    {}
                                                    (repeatedly (:width layer) u/new-id))
                                neuron-ids (keys new-neurons)
                                new-layer (assoc layer :neurons neuron-ids)
                                new-parents (if (empty? layers)
                                              parents
                                              (reduce (fn [conns neuron]
                                                        (assoc conns neuron (:neurons last-layer)))
                                                      parents
                                                      neuron-ids))]
                            {:layers (conj layers new-layer)
                             :neurons (merge neurons new-neurons)
                             :parents new-parents}))
                        {:layers []
                         :neurons {}
                         :parents {}}
                        desc)]
    (-> network
        (assoc :children (find-children (:parents network)))
        (assoc :learning-rate learning-rate)
        (merge opts))))

(defn get-neuron
  "Retrieve a neuron from the network."
  [network id]
  (get (:neurons network) id))

(defn get-parents
  "Retrieve a neurons parents from the network."
  [network id]
  (get (:parents network) id))

(defn get-children
  "Retrieve a neurons children from the network."
  [network id]
  (get (:children network) id))

(defn reset-network
  "Resets all the biases and weights in the network to random values."
  [])
