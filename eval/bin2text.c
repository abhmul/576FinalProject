//  This is a modification of distance.c, which was released by Google as part
//  of the word2vec code. Their original notice appears below.

//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f, *fo;
  char in_file[max_size], out_file[max_size];
  float len;
  long long words, size, a, b;
  char ch;
  float *M;
  char *vocab;
  if (argc < 3) {
    printf("Usage: ./bin2text <IN-FILE> <OUT-FILE>\nwhere IN-FILE contains word projections in the BINARY FORMAT, and OUT-FILE is the output text file name\n");
    return 0;
  }
  strcpy(in_file, argv[1]);
  strcpy(out_file, argv[2]);
  f = fopen(in_file, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    /* Don't normalize the vectors
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
    */
  }
  fclose(f);

  // Write the vectors to a text file
  fo = fopen(out_file, "w");
  fprintf(fo, "%lld %lld\n", words, size);
  for (a = 0; a < words; a++) {
    fprintf(fo, "%s ", &vocab[a * max_w]);
    for (b = 0; b < size; b++) fprintf(fo, "%f ", M[a * size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
  return 0;
}
