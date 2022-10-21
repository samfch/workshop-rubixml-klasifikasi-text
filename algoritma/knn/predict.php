<?php
require_once('vendor/autoload.php');
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('./knn/model-disimpan-knn.model')); //load model yang sudah disimpan sebelumnya

//contoh test data 
$judul_test = [
    ['Facility standards of vocational schools: Comparison of existing and modern facility designs'], //management
    ['Analysis of mediation effect of consumer satisfaction on the effect of service quality, price and consumer trust on consumer loyalty'], //management 
    ['The implementation of Multi-Objective Optimization on the Basis Of Ratio Analysis method to select the lecturer assistant working at computer laboratorium'], //management
    ['Pipe leakage detection system with artificial neural network'], //civil
    ['Smart Platform for Water Quality Monitoring System using Embedded Sensor with GSM Technology'] //civil
];

$data_test = new Unlabeled($judul_test);

// prediksi menggunakan data testing dari data diatas
$predictions = $estimator->proba($data_test);

//hasil testing menggunakan data testing dari data diatas
print_r($predictions);
