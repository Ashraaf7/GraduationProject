using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Web;
using System.Web.Mvc;
using CourseApp1.data;

namespace CourseApp1.Areas.Admin.Controllers
{
    public class RoadmapLinksController : Controller
    {
        private courses_dbEntities db = new courses_dbEntities();

        // GET: Admin/RoadmapLinks
        public ActionResult Index(int? id)
        {
            if (id == null)
            {
                return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
            }
            var roadmapLinks = db.RoadmapLinks.Include(r => r.Roadmap_courses).Where(r=>r.CourseId==id);
            ViewBag.courseid = id;
            return View(roadmapLinks.ToList());
        }

        // GET: Admin/RoadmapLinks/Details/5
        public ActionResult Details(int? id)
        {
            if (id == null)
            {
                return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
            }
            RoadmapLink roadmapLink = db.RoadmapLinks.Find(id);
            if (roadmapLink == null)
            {
                return HttpNotFound();
            }
            return View(roadmapLink);
        }

        // GET: Admin/RoadmapLinks/Create
        public ActionResult Create()
        {
            ViewBag.CourseId = new SelectList(db.Roadmap_courses, "Id", "Name");
            return View();
        }

        // POST: Admin/RoadmapLinks/Create
        // To protect from overposting attacks, enable the specific properties you want to bind to, for 
        // more details see https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Create([Bind(Include = "LinkId,CourseId,Link,,Name")] RoadmapLink roadmapLink,int? id)
        {
            if (id != null)
            {
                roadmapLink.CourseId = (int)id;
            }
            if (ModelState.IsValid)
            {
                db.RoadmapLinks.Add(roadmapLink);
                db.SaveChanges();
                return RedirectToAction("Index", new {id=id});
            }

            ViewBag.CourseId = new SelectList(db.Roadmap_courses, "Id", "Name", roadmapLink.CourseId);
            return View(roadmapLink);
        }

        // GET: Admin/RoadmapLinks/Edit/5
        public ActionResult Edit(int? id)
        {
            if (id == null)
            {
                return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
            }
            RoadmapLink roadmapLink = db.RoadmapLinks.Find(id);
            if (roadmapLink == null)
            {
                return HttpNotFound();
            }
            ViewBag.CourseId = new SelectList(db.Roadmap_courses, "Id", "Name", roadmapLink.CourseId);
            return View(roadmapLink);
        }

        // POST: Admin/RoadmapLinks/Edit/5
        // To protect from overposting attacks, enable the specific properties you want to bind to, for 
        // more details see https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Edit([Bind(Include = "LinkId,CourseId,Link,Name")] RoadmapLink roadmapLink)
        {
            if (ModelState.IsValid)
            {
                db.Entry(roadmapLink).State = EntityState.Modified;
                db.SaveChanges();
                return RedirectToAction("Index", new {id=roadmapLink.CourseId});
            }
            ViewBag.CourseId = new SelectList(db.Roadmap_courses, "Id", "Name", roadmapLink.CourseId);
            return View(roadmapLink);
        }

        // GET: Admin/RoadmapLinks/Delete/5
        public ActionResult Delete(int? id)
        {
            if (id == null)
            {
                return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
            }
            RoadmapLink roadmapLink = db.RoadmapLinks.Find(id);
            if (roadmapLink == null)
            {
                return HttpNotFound();
            }
            return View(roadmapLink);
        }

        // POST: Admin/RoadmapLinks/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public ActionResult DeleteConfirmed(int id)
        {
            RoadmapLink roadmapLink = db.RoadmapLinks.Find(id);
            db.RoadmapLinks.Remove(roadmapLink);
            db.SaveChanges();
            return RedirectToAction("Index", new { id = roadmapLink.CourseId });
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                db.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
