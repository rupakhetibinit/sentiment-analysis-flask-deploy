async function getData() {
  let response = await fetch(
    "https://api.twitter.com/2/tweets/search/recent?max_results=10&query=" +
      "nepal",
    {
      headers: {
        Authorization:
          "Bearer AAAAAAAAAAAAAAAAAAAAANULlgEAAAAAXOTsDrOEK4PI%2BbH896TomP%2FiGLE%3DPwpbppKix6JP0xNmzqjeo580M36evkckKmPgSIwvjokZs3T4ff",
        Cookie: "guest_id=v1%3A167656688150839753",
      },
    }
  );
  response = await response.json();
  // console.log(JSON.stringify(response));
  return response;
}

async function getSentiment() {}
async function sentiment(i) {
  let response = await fetch("http://127.0.0.1:5000/test", {
    body: i,
    headers: {
      "Content-Type": "application/json",
    },
    method: "POST",
  });
  response = await response.json();
  return response;
}

async function main() {
  console.time("new");

  let totalTweets = [];

  const data = await getData();
  try {
    await Promise.all(
      data.data.map(async (d) => {
        const contents = await sentiment(JSON.stringify(d));
        totalTweets.push(contents);
      })
    );
    console.log(JSON.stringify(totalTweets));
  } catch (error) {
    console.log(error);
  } finally {
    console.timeEnd("new");
  }
}
main()
  .then(() => console.log("done"))
  .catch((e) => console.log(e));
