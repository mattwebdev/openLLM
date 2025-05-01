package feeds

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/mmcdole/gofeed"
)

// RSSFeed represents an RSS feed source
type RSSFeed struct {
	URL       string
	Name      string
	LastFetch time.Time
}

// FeedManager manages multiple RSS feeds
type FeedManager struct {
	feeds  []RSSFeed
	parser *gofeed.Parser
	// Channel to send processed articles to the training system
	trainingChan chan<- string
}

// NewFeedManager creates a new feed manager
func NewFeedManager(trainingChan chan<- string) *FeedManager {
	return &FeedManager{
		parser:       gofeed.NewParser(),
		trainingChan: trainingChan,
		feeds: []RSSFeed{
			{
				URL:  "http://feeds.bbci.co.uk/news/rss.xml",
				Name: "BBC News",
			},
			{
				URL:  "https://feeds.bbci.co.uk/news/world/rss.xml",
				Name: "BBC World News",
			},
			{
				URL:  "https://feeds.bbci.co.uk/news/technology/rss.xml",
				Name: "BBC Technology News",
			},
			{
				URL:  "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
				Name: "BBC Science and Environment News",
			},
			{
				URL:  "https://feeds.bbci.co.uk/news/business/rss.xml",
				Name: "BBC Business News",
			},
		},
	}
}

// FetchAndProcess fetches and processes new articles from all feeds
func (fm *FeedManager) FetchAndProcess(ctx context.Context) error {
	for _, feed := range fm.feeds {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			if err := fm.processFeed(feed); err != nil {
				log.Printf("Error processing feed %s: %v", feed.Name, err)
				continue
			}
		}
	}
	return nil
}

// processFeed processes a single feed
func (fm *FeedManager) processFeed(feed RSSFeed) error {
	// Fetch the feed
	parsedFeed, err := fm.parser.ParseURL(feed.URL)
	if err != nil {
		return fmt.Errorf("failed to parse feed %s: %v", feed.URL, err)
	}

	// Process each item
	for _, item := range parsedFeed.Items {
		// Fetch the full article content
		content, err := fetchArticleContent(item.Link)
		if err != nil {
			log.Printf("Failed to fetch article content from %s: %v", item.Link, err)
			// Use description if full content fetch fails
			content = item.Description
		}

		// Send to training channel
		select {
		case fm.trainingChan <- content:
			// Successfully sent to training
		default:
			// Channel is full, skip this article
			log.Printf("Training channel full, skipping article: %s", item.Title)
		}
	}

	return nil
}

// fetchArticleContent fetches and parses the full article content
func fetchArticleContent(url string) (string, error) {
	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: 10 * time.Second,
	}

	// Fetch the article page
	resp, err := client.Get(url)
	if err != nil {
		return "", fmt.Errorf("failed to fetch article: %v", err)
	}
	defer resp.Body.Close()

	// Parse the HTML
	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to parse HTML: %v", err)
	}

	// Extract the main content
	var content strings.Builder

	// Try different selectors for main content
	selectors := []string{
		"article[role='article']",        // Main article content
		".story-body__inner",             // Story body
		".vxp-media__body",               // Video content
		".ssrcss-1f3bvyz-ArticleWrapper", // New BBC layout
	}

	for _, selector := range selectors {
		doc.Find(selector).Each(func(i int, s *goquery.Selection) {
			// Remove unwanted elements
			s.Find("script, style, iframe, .advertisement, .related-content").Remove()

			// Get text content
			text := s.Text()
			text = strings.TrimSpace(text)
			if text != "" {
				content.WriteString(text)
				content.WriteString("\n\n")
			}
		})
	}

	// If no content found with selectors, fall back to description
	if content.Len() == 0 {
		return "", fmt.Errorf("no content found with selectors")
	}

	return content.String(), nil
}

// StartBackgroundFetching starts a background goroutine to fetch feeds periodically
func (fm *FeedManager) StartBackgroundFetching(ctx context.Context, interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if err := fm.FetchAndProcess(ctx); err != nil {
					log.Printf("Error in background feed fetching: %v", err)
				}
			}
		}
	}()
}
