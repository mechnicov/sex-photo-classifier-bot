## Sex photo classifier Telegram Bot

### Launch

Clone this repo

Set up your credentials, please see example:

```
$ cp .env.example .env
```

Build docker image:

```
$ docker build . -t sex_photo_classifier
```

To run:

```
$ docker run --name sex_photo_classifier -d sex_photo_classifier
```

The first launch can be lengthy due to the machine learning process

You can send photo to bot. Bot will try to guess sex. Also you can leave feedback

### Example of usage

![Example of usage](https://github.com/mechnicov/sex-photo-classifier-bot/blob/master/example.jpg?raw=true)
