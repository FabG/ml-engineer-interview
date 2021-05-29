# System Design Interview
The technical portion of the interview is more for assessing the System Architecture experience of a candidate.
We will be expecting you to visualize the candidate's thoughts and design on the whiteboard with block diagrams and rough outlines of some class definitions and API contracts.


### System Design Open Questions

- Describe a system / product / app you or your team built.

- How did you evaluate the design of your system?

- How did you test performance and scalability?

- Did you have to iterate on the design?

- Looking back, what would you have done differently?



### System Design Specific Questions
#### What is an in-memory datastore and when would you use it? What technologies exist?

From [AWS](https://aws.amazon.com/nosql/in-memory/): An in-memory database is a type of purpose-built database that relies primarily on memory for data storage, in contrast to databases that store data on disk or SSDs. In-memory databases are designed to attain minimal response time by eliminating the need to access disks. Because all data is stored and managed exclusively in main memory, it is at risk of being lost upon a process or server failure. In-memory databases can persist data on disks by storing each operation in a log or by taking snapshots.

In-memory databases are ideal for applications that require microsecond response times and can have large spikes in traffic coming at any time such as gaming leaderboards, session stores, and real-time analytics.

Use cases
- **Real-time bidding**: Real-time bidding refers to the buying and selling of online ad impressions. Usually the bid has to be made while the user is loading a webpage, in 100-120 milliseconds and sometimes as little as 50 milliseconds. During this time period, real-time bidding applications request bids from all buyers for the ad spot, select a winning bid based on multiple criteria, display the bid, and collect post ad-display information. In-memory databases are ideal choices for ingesting, processing, and analyzing real-time data with submillisecond latency.  

- **Gaming leaderboards**: A relative gaming leaderboard shows a gamer's position relative to other players of a similar rank. A relative gaming leaderboard can help to build engagement among players and meanwhile keep gamers from becoming demotivated when compared only to top players. For a game with millions of players, in-memory databases can deliver sorting results quickly and keep the leaderboard updated in real time.

- **Caching**: A cache is a high-speed data storage layer which stores a subset of data, typically transient in nature, so that future requests for that data are served up faster than is possible by accessing the data’s primary storage location. Caching allows you to efficiently reuse previously retrieved or computed data. The data in a cache is generally stored in fast access hardware such as RAM (Random-access memory) and may also be used in correlation with a software component. A cache's primary purpose is to increase data retrieval performance by reducing the need to access the underlying slower storage layer.

In-memory databases on AWS
- **Amazon Elasticache for Redis**: Amazon ElastiCache for Redis is a blazing fast in-memory data store that provides submillisecond latency to power internet-scale, real-time applications. Developers can use ElastiCache for Redis as an in-memory nonrelational database. The ElastiCache for Redis cluster configuration supports up to 15 shards and enables customers to run Redis workloads with up to 6.1 TB of in-memory capacity in a single cluster. ElastiCache for Redis also provides the ability to add and remove shards from a running cluster. You can dynamically scale out and even scale in your Redis cluster workloads to adapt to changes in demand.

- **Amazon ElastiCache for Memcached**: Amazon ElastiCache for Memcached is a Memcached-compatible in-memory key-value store service that can be used as a cache or a data store. It delivers the performance, ease-of-use, and simplicity of Memcached. ElastiCache for Memcached is fully managed, scalable, and secure - making it an ideal candidate for use cases where frequently accessed data must be in-memory. It is a popular choice for use cases such as Web, Mobile Apps, Gaming, Ad-Tech, and E-Commerce.

Choosing between Redis and Memcached
Redis and Memcached are popular, open-source, in-memory data stores. Although they are both easy to use and offer high performance, there are important differences to consider when choosing an engine.
- **Memcached** is designed for simplicity while **Redis** offers a rich set of features that make it effective for a wide range of use cases. Understand your requirements and what each engine offers to decide which solution better meets your needs.
- **Memcached** is multithreaded but Redis is not. Memcached can make use of multiple processing cores. This means that you can handle more operations by scaling up compute capacity.
- **Redis** lets you create multiple replicas of a Redis primary. This allows you to scale database reads and to have highly available clusters. Memcached does not.
- **Redis** supports transactions which let you execute a group of commands as an isolated and atomic operation. Memcached does not.
- **Redis** also supports Pub/Sub messaging with pattern matching which you can use for high performance chat rooms, real-time comment streams, social media feeds, and server intercommunication.
- **Redis** has purpose-built commands for working with real-time geospatial data at scale. You can perform operations like finding the distance between two elements (for example people or places) and finding all elements within a given distance of a point.


#### What are the differences between SOAP and REST APIs
- SOAP stands for Simple Object Access Protocol whereas REST stands for Representational State Transfer.
- SOAP is a protocol whereas REST is an architectural pattern.
- SOAP uses service interfaces to expose its functionality to client applications while REST uses Uniform Service locators to access to the components on the hardware device.  
- SOAP was designed with a specification. It includes a WSDL file which has the required information on what the web service does in addition to the location of the web service.
- SOAP needs more bandwidth for its usage whereas REST doesn’t need much bandwidth.
- Comparing SOAP vs REST API, SOAP only works with XML formats whereas REST work with plain text, XML, HTML and JSON.
- SOAP cannot make use of REST whereas REST can make use of SOAP.

#### What is Base64?
Base64 is a group of binary-to-text encoding schemes that represent binary data in an ASCII string format by translating it into a radix-64 representation.
Base64 encoding schemes are commonly used when there is a need to encode binary data that needs to be stored and transferred over media that are designed to deal with ASCII. This is to ensure that the data remain intact without modification during transport. Base64 is commonly used in a number of applications including email via MIME, and storing complex data in XML.

One common application of base64 encoding on the web is to encode binary data  so it can be included in a `data: URL`.
Examples:
- `data:,Hello%2C%20World%21`
The text/plain data Hello, World!. Note how the comma is percent-encoded as %2C, and the space character as %20.
- `data:text/plain;base64,SGVsbG8sIFdvcmxkIQ==`
base64-encoded version of the above
- `data:text/html,%3Ch1%3EHello%2C%20World%21%3C%2Fh1%3E`
An HTML document with `<h1>Hello, World!</h1>`

Encoded size increase:

Each Base64 digit represents exactly 6 bits of data. So, three 8-bits bytes of the input string/binary file (3×8 bits = 24 bits) can be represented by four 6-bit Base64 digits (4×6 = 24 bits).

This means that the Base64 version of a string or file will be at most 133% the size of its source (a ~33% increase). The increase may be larger if the encoded data is small. For example, the string "a" with length === 1 gets encoded to "YQ==" with length === 4 — a 300% increase.


#### What is MD5?
The MD5 message-digest algorithm is a widely used hash function producing a 128-bit hash value. Although MD5 was initially designed to be used as a cryptographic hash function, it has been found to suffer from extensive vulnerabilities. It can still be used as a checksum to verify data integrity, but only against unintentional corruption. It remains suitable for other non-cryptographic purposes, for example for determining the partition for a particular key in a partitioned database.
One basic requirement of any cryptographic hash function is that it should be computationally infeasible to find two distinct messages that hash to the same value. MD5 fails this requirement catastrophically; such collisions can be found in seconds on an ordinary home computer.
