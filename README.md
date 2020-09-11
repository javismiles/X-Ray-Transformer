# X-Ray-Transformer
# The X-Ray Transformer Infographic
The transformer arquitecture has produced a revolution in the NLP field and in deep learning. A multitude of applications are benefiting from the capacity of this models to process sequences in parallel while achieving a deeper understanding of their context through the attention mechanism they implement.

Understanding how the transformer processes sequences can be a bit challenging at first. When tackling a complex model, many people, me included, like to study how the computations of the model change the shapes of the tensors that travel through it

To that purpose, I created the X-Ray Transformer infographic, that allows you to make the journey from the beginning to the end of the transformer's computations in both the training and inference phases. Its objective is to achieve a quick and deep understanding of the inner computations of a transformer model through the analysis and exploration of a single visual asset.

When looking at this infographic, the first thing to consider, shown at the bottom right of the graphic, is the colors that denote different important stages.

Light Blue denotes the training phase.

Light Green denotes the inference phase.

Purple indicates the Encoding stage, and within the encoder purple modules belong to the training phase and green ones to the inference phase

Dark Red indicates the Decoding stage. Within the decoder, purple modules indicate encoder data, dark red indicates decoder data, and green modules express as usual the inference stage.

Once clarified the color codes, the next is to notice the pink circles with numbers inside them. Those help us see the general path of execution, first moving through the encoder and then the decoder.

The two large arrows on both sides remind of some key stages of the execution of both the encoder and decoder phases. 

To generate this infographic, I used a small transformer model that produces a chatbot. The chatbot is trained with pairs of questions and answers. This specific model is trained on questions and answers related to movies and series, specially science fiction ones. Examples of questions and answers:

"What's your favourite character in The Expanse series?" : "Naomi Nagata definitely!"
"What's your favourite character in Battlestar Galactica?" : "Kara Thrace, she is great"

Below the title of the infographic we can review the most important parameters to consider when studying the shapes of the computations. This small model of a transformer is trained with a batch size of 8. The model has 4 yeads in its multi-head attention part, and there are 3 encoder layers and 3 decoder layers.
Also the size of the output vocabulary of the model is 950, and the embedding size used across the model is 32.

We begin the journey on the bottom left of the infographic as we begin to train the model. We get a batch (size of 8). The batch is composed of 8 sequences. Those 8 sequences are padded as necessary so that they all have the same length, in this case 10.

These sequences have been at the beginning of the process tokenized and then numericalized to prepare them to be ingested by the model. By the time the training loop extracts a new batch, the sequences are numericalized and structured ni a tensor of dimensions 8x10 (BS x SeqLen).

Next we need to create a mask that will help us prevent that the additional padding elements in the sequance be taken into account by the attention mechanisms. 

Now we have to create our embeddings, so we send the 8x10 tensor to the embed module and get back an 8x10x32 tensor because our embedding size is 32.

To that, we add the result of the positional encoding module, which will help the model take into account the difference in positioning across the sequence.

The first layer of our encoder is ready to ingest this 8x10x32 tensor. The first thing the encoder does is to create three copies of this tensor to produce the Q, K and V elements of the model, that is, the query, keys and values.

These 3 tensors are passed through a linear layer first and then we arrive to the point of having to split into our 4 heads. Using 4 heads will allow the attention mechanism to interpret the sequences from different perspectives.

Computationally we can prepare this stage in two simple steps. First we can reconfigure the tensor to split the embedding size dimension, 32, into two dimensions, 4 and 8. 4 is the number of heads. And 8? 8 is what we call dimK, which is equal to the Embedding dimensions / number of heads 32 / 4 = 8.

And now we do a transpose operation to position the dimension with the number of heads after the batch size one. That produces the new shapes: 8 x 4 x 10 x 8. What this shape tells us is: for each element of the batch, we will have 4 heads. And each of those heads will have a 10(sequence length) x 8(dimK) structure within.

Out objective now is to calculate the attention scores. We are in this case performing what we call self-attention. That is how much attention should different parts of our sequence pay to different parts of itself?
To find this out we will multiply the Query and Keys tensors. To multiply them we need to transpose the second half of the K tensor. Once the K tensor has been transposed, now we have two shapes that we can multiple:
8x4x10x8 times 8x4x8x10.  Notice that we are really multiplying are the two last dimensions: 10x8 times 8x10. This is going to produce the attention scores tensor which will have the shape of: 8x4x10x10.
These are our self-attention scores. For each element of the batch, and for each of the 4 heads, we have a 10x10 matrix, which expresses how much attention each of the parts of our sequence should pay to each of the parts of the same sequence.

The next thing we will do is to apply a mask. This is because remember that we made sure that all the sequences in the batch would have the very same length. And to do that we had to add padding tokens to the sequences that were shorter than the largest one. We now should mask (make really small or negative) those parts of the sequence that had the padding token. So we apply the mask and eliminate the influence of the parts of the sequences that correspond to the padding tokens.

Now we will apply a softmax module to the 10x10 matrix, so that all the numbers of each row sum to 1, converting each row into a probability distribution. Those are our soft self attention scores. For each sequence of each batch and within each head, how strong is the connection between each part of that sequence and each part of itself, and with the sum of all the influences on each part of the sequence adding to one.

Now that we have the attention scores, we should apply them to the values, to the V tensor. Our attention scores have the shape of 8x4x10x10. And our V tensor has the shape of 8x4x10x8. Remember that we are really multiplying the last 2 dimensions, so we are multiplying 10x10 times 10x8. This produces a new tensor of dimension 8x4x10x8.

At this stage, we have concluded the self-attention stage. We found out the attention scores by multiplying the queries and the keys. And then applied those attention scores to the values to obtain the final attention tensor. 

It's the moment to unify the 4 heads into one. To do that, we do the inverse of before, combining tranposition and reconfiguration to obtain a new shape of 8x10x32.

After passing it through a linear layer, we arrive to our first skip connection. We will add our current tensor to original one that entered the encoder layer. And then we will apply layer normalization to keep the data in a good range.

Next, we pass our 8x10x32 tensor through a feedforward layer and then apply another skip connection, adding the resulting tensor to the one that entered the feed forward layer.

Wonderful! That was one layer of the encoder. Now the very same computations will be applied x number of times corresponding to whatever x number of layers we have.

Notice that the tensor that entered the encoder layer and the one that exits the encoder layer have the very same shape: 8x10x32. That's why we can chain as many encoder layers as we like one after the other.

Once we arrive to the final encoder layer, we obtain our final 8x10x32 tensor. This is a very important tensor which will connect to the encoder-decoder attention mechanism of the decoder to provide the keys and values that will interact with the questions from the decoder. 

But before we go there, let's move to the next step. The bottom part of the decoder.
