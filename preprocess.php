<?php

require_once('vendor/autoload.php');

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Transformers\RegexFilter;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\TextNormalizer;
use Rubix\ML\Transformers\StopWordFilter;
use Rubix\ML\Tokenizers\WordStemmer;
use Rubix\ML\Tokenizers\Word;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Transformers\LambdaFunction;
use \Sastrawi\Stemmer\StemmerFactory;

// Load dataset dari csv
$dataset = Labeled::fromIterator(new CSV('judul-indo.csv'));

// cek dataset
// print_r($dataset);

// membagi dataset menjadi data testing dan training
[$training, $testing] = $dataset->stratifiedSplit(0.8);

// fungsi untuk stemmer bahasa inggris dengan fungsi bawaan rubixml
$stemmeringgris = function (&$sample, $offset, $context) {
    $stemmer = new WordStemmer('english');
    $sample[0] = implode($stemmer->tokenize($sample[0]));
};

//fungsi stemmer bahasa indonesia dengan library sastrawi
$stemmerindo = function (&$sample, $offset, $context) {
    $stemmerFactory = new StemmerFactory();
    $stemmer  = $stemmerFactory->createStemmer();
    $sample[0] = $stemmer->stem($sample[0]);
};

// pipeline untuk preprocessing dan klasifikasi
$estimator = new Pipeline([
    //PREPROCESSING
    new RegexFilter([RegexFilter::URL,RegexFilter::EXTRA_CHARACTERS]), //text cleaning
    new TextNormalizer(false), //case folding - lower case
    new StopWordFilter(['i', 'me', 'my', 'to', 'in', 'as', 'not']), //stopword
    // new LambdaFunction($stemmeringgris, 'stemmer'), //stemming inggris
    new LambdaFunction($stemmerindo, 'stemmer'), //stemming indonesia
    new WordCountVectorizer(), //tokenizer

    // ALGORITMA KLASIFIKASI
], new KNearestNeighbors(3), true);

//proses training
$estimator->train($training);

// cek apakah proses training berhasil
// print_r($estimator->trained());

// testing menggunakan data testing dari dataset
$predictions = $estimator->predict($testing);

//hasil testing menggunakan data testing dari dataset
print_r($predictions);

// validasi, confusion matrix, akurasi, dll
$report = new MulticlassBreakdown();

$results = $report->generate($predictions, $testing->labels());

echo $results;

/**
 * CATATAN:
 * - pilih contoh data judul.csv atau judul-indo.csv
 * - pilihlah stemmer untuk bahasa inggris dan bahasa indonesia, atau pilih dua2nya
 * - anda bisa membandingkan bandingkan hasil validasi dengan stemmer dan tanpa stemmer
 */