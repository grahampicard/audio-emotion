from spotipy.oauth2 import SpotifyClientCredentials

import json
import os
import requests
import spotipy


class track_metadata_handler:
    def __init__(self, lastfm_credentials=None, spotipy_client_id=None,
                 spotipy_client_secret=None):
        """
        This handler assumes you have the following environmental varaibles

        SPOTIPY_CLIENT_ID
        SPOTIPY_CLIENT_SECRET
        LASTFM_SECRET

        If you don't, complete the following
        
        Windows:    Open start, and load these values as environmental variables 
                    and restart.

        Mac/Linux:  load variables into an .env file (I can provide) and run
                    `source .env` before running the script
        """
        self.spotify_credentials = SpotifyClientCredentials()
        self.lastfm_credentials = None

        if (spotipy_client_id != None) & (spotipy_client_secret != None):
            self.spotify_credentials = SpotifyClientCredentials(
                client_id=spotipy_client_id,
                client_secret=spotipy_client_secret
            )

        if lastfm_credentials == None:
            self.lastfm_credentials = os.environ['LASTFM_SECRET']

        else:
            self.lastfm_credentials = lastfm_credentials

        # create spotify handler
        self.spotify_handler = spotipy.Spotify(
            client_credentials_manager=self.spotify_credentials
        )

    def get_track_metadata(self, track=None, artist=None, album=None,
                           spotify=True, lastfm=True):
        """ Supply at least the track. The function will automatically format the 
            Spotify query string.            
        """
        
        result = None
        spotify_data = None
        last_fm_data = None
        
        if spotify is True:
            spotify_query_string = ""
            query_type = ""

            if track:
                spotify_query_string += track
                query_type += "track"

            if artist:
                spotify_query_string += " " + artist
                query_type += ",artist"
            
            if album:
                spotify_query_string += " " + album
                query_type += ",album"
        
            spotify_data = self._get_spotify_track_info(spotify_query_string,
                                                        query_type)

        if lastfm is True:
            lastfm_artist = self._lastfm_prepare(artist)
            lastfm_track = self._lastfm_prepare(track)
            last_fm_data = self._get_lastfm_tags(lastfm_artist, lastfm_track)

        if spotify and lastfm:
            result = spotify_data
            result["tags"] = last_fm_data
        elif spotify:
            result = spotify_data
        elif lastfm:
            result = last_fm_data
        else:
            raise EOFError

        return result

    def _get_spotify_track_info(self, query, query_type):
        """ Valid search types:
            album, artist, playlist, track, show, episode

            e.g. type='track,artist'
        """

        # get the song's spotify ID
        resp = self.spotify_handler.search(query, type=query_type)
        top_match = resp["tracks"]["items"][0]
        data = {key: top_match[key] for key in ["id", "name"]}
        data["artists"] = [x["name"] for x in top_match["artists"]]

        # find the audio features for the song
        features = self.spotify_handler.audio_features(data["id"])[0]
        
        ####
        # Add any additional spotify track data here
        ####

        result = {**data, **features}

        # merge and return
        return result

    def _lastfm_prepare(self, string):
        if " " in string:
            string = string.replace(" ", "+")
        return string

    def _get_lastfm_tags(self, artist, track, cutoff=3):
        url = f"http://ws.audioscrobbler.com/2.0/?method=track.gettoptags&artist={artist}&track={track}&api_key={self.lastfm_credentials}&format=json"
        tags = json.loads(requests.get(url).text)
        tags = tags["toptags"]["tag"]
        tags = [x for x in tags if x["count"] >= cutoff]
        return tags
